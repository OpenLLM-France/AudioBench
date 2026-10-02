"""Re-apply ASR post-processing to already-computed predictions, without re-running inference.

For every ASR result file under a model's results directory (identified by
``metadata.task_type == "ASR"``), this:

  1. backs up the prediction file  ``<dataset>.json``       -> ``<dataset>.json.deprecated``
     and the score file            ``<dataset>_score.json`` -> ``..._score.json.deprecated``
     (once each -- the .deprecated always holds the original),
  2. rewrites each ``model_prediction`` through the OUTPUT post-processor chosen by the model
     folder name -- Qwen3-Omni: quote/intro recovery; Audio-Flamingo: description-preamble
     strip; Phi-4: "Spoken text:" label strip (override with ``--postproc``), and
  3. recomputes the WER ``_score.json`` from the cleaned predictions (same normalization and
     metric as the evaluation pipeline -- no audio, no model, no config needed).

Note: only OUTPUT fixes are rescorable here. A model whose fix is an input-instruction rewrite
(Phi-4's file-reference neutralization) must be RE-RUN to get that part; rescoring only re-applies
its output cleanup.

Usage:
    python -m audio_bench.rescore_asr_postprocess results/qwen3_omni_30b_instruct
    python -m audio_bench.rescore_asr_postprocess results/audio_flamingo_3 --dry_run
    python -m audio_bench.rescore_asr_postprocess results/phi_4_multimodal_instruct
    python -m audio_bench.rescore_asr_postprocess results/<model> --restore
"""

import glob
import json
import os
import shutil

import fire

from audio_bench.asr_postprocess import (
    postprocess_asr_prediction,
    strip_audio_description_preamble,
    strip_label_prefix,
)

# Which output post-processor to re-apply, chosen by the model folder name. Only the OUTPUT
# post-processors are rescorable without re-inference; a model whose fix is an input-instruction
# rewrite (e.g. Phi-4's file-reference neutralization) must be re-run to get that part -- here we
# can only re-apply its output cleanup ("Spoken text:" stripping).
_POSTPROC_BY_MODEL = (
    ("audio_flamingo", strip_audio_description_preamble),
    ("phi_4", strip_label_prefix),
    ("phi4", strip_label_prefix),
    ("qwen3_omni", postprocess_asr_prediction),
)


def _select_postproc(model_dir, override=None):
    if override:
        return {"quotes": postprocess_asr_prediction, "label": strip_label_prefix,
                "flamingo": strip_audio_description_preamble}[override]
    name = os.path.basename(os.path.normpath(model_dir)).lower()
    for key, fn in _POSTPROC_BY_MODEL:
        if key in name:
            return fn
    return postprocess_asr_prediction  # default: Qwen3-style quote/intro recovery


def _iter_asr_prediction_files(model_dir):
    for path in sorted(glob.glob(os.path.join(model_dir, "**", "*.json"), recursive=True)):
        if path.endswith("_score.json") or path.endswith(".deprecated"):
            continue
        try:
            data = json.loads(open(path, encoding="utf-8").read())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and data.get("metadata", {}).get("task_type") == "ASR":
            yield path


def _compute_wer(predictions):
    """Return the pipeline's WER result dict for a list of {reference, model_prediction}."""
    from audio_bench.scoring_src.metrics import (
        compute_wer,
        get_predictions_and_references_lists,
    )
    preds, refs = get_predictions_and_references_lists(predictions)
    return compute_wer(refs, preds)


def _load(path):
    return json.loads(open(path, encoding="utf-8").read())


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def main(model_dir, dry_run=False, restore=False, postproc=None):
    files = list(_iter_asr_prediction_files(model_dir))
    if not files:
        print(f"No ASR prediction files found under {model_dir}")
        return

    postprocess = _select_postproc(model_dir, postproc)
    print(f"{'RESTORE' if restore else ('DRY-RUN' if dry_run else 'APPLY')}: "
          f"{len(files)} ASR dataset(s) under {model_dir} "
          f"[postproc: {postprocess.__name__}]\n")

    total_changed = total_preds = 0
    for pred_path in files:
        rel = os.path.relpath(pred_path, model_dir)
        score_path = pred_path[:-len(".json")] + "_score.json"
        dep_pred, dep_score = pred_path + ".deprecated", score_path + ".deprecated"

        if restore:
            restored = []
            for live, dep in ((pred_path, dep_pred), (score_path, dep_score)):
                if os.path.exists(dep):
                    shutil.move(dep, live)
                    restored.append(os.path.basename(live))
            print(f"  {rel}: restored {restored or 'nothing (no .deprecated)'}")
            continue

        # Always post-process from the ORIGINAL predictions (the .deprecated if present).
        src = dep_pred if os.path.exists(dep_pred) else pred_path
        data = _load(src)
        preds = data["predictions"]

        n_changed = sum(
            1 for p in preds
            if postprocess(p["model_prediction"]) != p["model_prediction"]
        )
        for p in preds:
            p["model_prediction"] = postprocess(p["model_prediction"])

        # WER before/after (best-effort: needs jiwer).
        old_w = new_w = None
        try:
            if os.path.exists(score_path):
                sw = _load(score_path).get("wer")
                old_w = sw.get("score") if isinstance(sw, dict) else sw
            new_w = _compute_wer(preds)["wer"]["score"]
        except Exception as e:  # jiwer missing, or an odd score file
            wer_note = f"(WER n/a: {type(e).__name__})"
        else:
            def pct(x):
                return f"{x * 100:.1f}" if isinstance(x, (int, float)) else "?"
            wer_note = f"WER {pct(old_w)} -> {pct(new_w)}"

        print(f"  {rel}: changed {n_changed}/{len(preds)}  {wer_note}")
        total_changed += n_changed
        total_preds += len(preds)

        if dry_run:
            continue

        # Persist: back up originals once, write cleaned predictions + recomputed score.
        if not os.path.exists(dep_pred):
            shutil.copy2(pred_path, dep_pred)
        _dump(data, pred_path)

        if os.path.exists(score_path):
            if not os.path.exists(dep_score):
                shutil.copy2(score_path, dep_score)
            score = _load(score_path)
            wer_result = _compute_wer(preds)
            score["wer"] = wer_result["wer"]
            score["details"] = wer_result["details"][:20]  # match pipeline truncation
            _dump(score, score_path)

    print(f"\nTotal: {total_changed}/{total_preds} predictions changed"
          f"{' (dry-run, nothing written)' if dry_run else ''}.")


if __name__ == "__main__":
    fire.Fire(main)
