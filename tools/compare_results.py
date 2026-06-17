#!/usr/bin/env python3
"""Visualize one or compare N models' predictions/scores on a dataset.

Reads files written by `audio_bench.main_evaluate`:
  <root>/<model>/<LANG>/<dataset>.json          # predictions
  <root>/<model>/<LANG>/<dataset>_score.json    # per-metric scores + all_scores

Generates a self-contained HTML report (and a terminal summary).

Examples:
  # Visualize one model on one dataset
  python tools/compare_results.py results --dataset alpaca_audio -m phi_4_multimodal_instruct --language EN

  # Compare three models, sort samples by score spread
  python tools/compare_results.py results --dataset alpaca_audio \\
      -m phi_4_multimodal_instruct audio_flamingo_3 qwen2_audio_7b_instruct \\
      --language EN --sort-by-diff

  # Generate one report per dataset matching a task (e.g. "ASR", "QUESTION ANSWERING")
  python tools/compare_results.py results --task ASR \\
      -m phi_4_multimodal_instruct audio_flamingo_3 --language IT
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Optional


PRED_PALETTE = [
    ("#eef3ff", "#cdd9f5"),  # blue
    ("#fff5e6", "#f5d8a8"),  # orange
    ("#f3eaff", "#dccaf2"),  # purple
    ("#eafbf3", "#bfe7d2"),  # green
    ("#fde9ef", "#f3c4d1"),  # pink
    ("#fbf6d8", "#e6dc9b"),  # yellow
    ("#e6f6fb", "#b9dde9"),  # cyan
]


def find_files(root: Path, model: str, dataset: str, language: Optional[str]):
    """Return (predictions_path, score_path, resolved_language) or (None, None, None)."""
    model_dir = root / model
    if not model_dir.is_dir():
        print(f"[error] model dir not found: {model_dir}", file=sys.stderr)
        return None, None, None

    candidates = [language] if language else [d.name for d in model_dir.iterdir() if d.is_dir()]
    candidates = candidates + [None]  # also try flat layout

    for lang in candidates:
        base = model_dir / lang if lang else model_dir
        pred = base / f"{dataset}.json"
        score = base / f"{dataset}_score.json"
        if pred.is_file() or score.is_file():
            return (pred if pred.is_file() else None,
                    score if score.is_file() else None,
                    lang)
    return None, None, None


def load_json(path: Optional[Path]):
    if path is None:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"[warn] cannot read {path}: {e}", file=sys.stderr)
        return None


def extract_metric_scores(score_data):
    if not isinstance(score_data, dict):
        return {}
    out = {}
    for k, v in score_data.items():
        if isinstance(v, dict) and ("score" in v or "all_scores" in v):
            out[k] = v
    return out


def fmt_score(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_summary(label, score_data):
    if not score_data:
        print(f"[{label}] no score file")
        return
    metrics = extract_metric_scores(score_data)
    n = score_data.get("number_of_samples")
    print(f"\n[{label}] dataset={score_data.get('dataset_name')} "
          f"language={score_data.get('language')} samples={n}")
    for m, v in metrics.items():
        print(f"   {m:25s} score={fmt_score(v.get('score'))} "
              f"std={fmt_score(v.get('std'))} n_scores={len(v.get('all_scores') or [])}")


# ---------- HTML rendering ----------

def build_css():
    base = """
body { font-family: -apple-system, Segoe UI, sans-serif; margin: 1.5em; color:#222; }
h1 { margin-bottom: 0.2em; }
.meta { color:#666; margin-bottom:1em; }
table.summary { border-collapse: collapse; margin-bottom:1.5em; }
table.summary th, table.summary td { border:1px solid #ddd; padding:4px 10px; text-align:left; }
table.summary th { background:#f4f4f4; }
.sample { border:1px solid #ddd; border-radius:6px; padding:10px 14px; margin:10px 0; background:#fafafa; }
.sample h3 { margin:0 0 6px 0; font-size:14px; color:#444; }
.row { display:flex; gap:14px; margin-top:6px; flex-wrap:wrap; }
.col { flex:1 1 220px; min-width:0; }
.col h4 { margin:0 0 4px 0; font-size:12px; text-transform:uppercase; color:#888;
         white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.modelname { font-size:12px; text-transform:uppercase; color:#888; font-weight:600;
         word-break:break-word; margin:0 0 3px 0; }
.scoreline { display:flex; flex-wrap:wrap; gap:4px; margin:0 0 4px 0; }
.scoreline .score { font-size:12px; padding:1px 6px; border-radius:4px; background:#fff;
         border:1px solid #eee; font-weight:bold; }
.text { white-space:pre-wrap; word-wrap:break-word; background:#fff; border:1px solid #eee;
        border-radius:4px; padding:6px 8px; font-size:13px; max-height:340px; overflow:auto; }
.ref { background:#eef7ee; border-color:#cfe6cf; }
.score { font-weight:bold; }
.score.good { color:#2a7a2a; }
.score.bad  { color:#a83232; }
.score.mid  { color:#a87a32; }
.diff { padding:2px 6px; border-radius:4px; font-size:12px; background:#eee; color:#444; margin-left:6px; }
.controls { margin: 0.5em 0 1em; font-size:13px; }
.instr { color:#555; font-style:italic; font-size:13px; margin:2px 0 6px 0; }
"""
    palette = "\n".join(
        f".pred{i+1} {{ background:{bg}; border-color:{bd}; }}"
        for i, (bg, bd) in enumerate(PRED_PALETTE)
    )
    return base + palette


def score_class(s, max_s=5.0):
    if s is None:
        return ""
    try:
        s = float(s)
    except Exception:
        return ""
    if s < 0:
        return "bad"
    ratio = s / max_s if max_s else 0
    if ratio >= 0.7:
        return "good"
    if ratio >= 0.4:
        return "mid"
    return "bad"


def render_html(args, models, preds_list, scores_list):
    """models: list[str], preds_list/scores_list: aligned list of dicts (or None)."""
    metrics_per_model = [extract_metric_scores(s) if s else {} for s in scores_list]
    primary_metric = next(
        (m for mm in metrics_per_model for m in mm),
        None,
    )
    preds = [(p or {}).get("predictions") or [] for p in preds_list]
    all_scores = [
        (mm.get(primary_metric, {}) or {}).get("all_scores") or []
        for mm in metrics_per_model
    ]

    # Per-sample scores for every metric, per model: per_model_metric_scores[idx] = {metric: [scores]}
    # metric_max scales the good/mid/bad coloring per metric (metrics have different ranges).
    per_model_metric_scores = []
    metric_max = {}
    for mm in metrics_per_model:
        d = {}
        for met, v in mm.items():
            scores = (v or {}).get("all_scores") or []
            d[met] = scores
            mx = max((x for x in scores if isinstance(x, (int, float))), default=0)
            metric_max[met] = max(metric_max.get(met, 0), mx, 1)
        per_model_metric_scores.append(d)

    n = max((len(p) for p in preds), default=0)

    indices = list(range(n))
    if args.sort_by_diff and len(models) >= 2:
        def spread(i):
            vals = [s[i] for s in all_scores
                    if i < len(s) and isinstance(s[i], (int, float))]
            if len(vals) < 2:
                return -1e9
            return max(vals) - min(vals)
        indices.sort(key=spread, reverse=True)
    if args.limit and args.limit > 0:
        indices = indices[:args.limit]

    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>{html.escape(args.dataset)} — {html.escape(' vs '.join(models))}",
             "</title><style>", build_css(), "</style></head><body>"]

    parts.append(f"<h1>{html.escape(args.dataset)}</h1>")
    parts.append(f"<div class='meta'>language={html.escape(str(args.language or '?'))} "
                 f"results_root={html.escape(str(args.root))} "
                 f"models={html.escape(', '.join(models))}</div>")

    # summary table
    parts.append("<table class='summary'><tr><th>Metric</th>")
    for m in models:
        parts.append(f"<th>{html.escape(m)}</th>")
    parts.append("</tr>")

    all_metrics = list(dict.fromkeys(m for mm in metrics_per_model for m in mm))
    for met in all_metrics:
        row = [f"<tr><td>{html.escape(met)}</td>"]
        for mm in metrics_per_model:
            v = mm.get(met, {}).get("score")
            row.append(f"<td>{fmt_score(v)}</td>")
        row.append("</tr>")
        parts.append("".join(row))
    parts.append("</table>")

    note = ""
    if args.sort_by_diff and len(models) >= 2:
        note = f" (sorted by score spread of {html.escape(primary_metric or '')})"
    parts.append(f"<div class='controls'>Showing {len(indices)} / {n} samples{note}. "
                 f"Primary metric: <b>{html.escape(primary_metric or '—')}</b>.</div>")

    def header_block(name, metric_scores):
        """Model name on its own line, then a wrapping row of per-metric score chips."""
        out = f"<div class='modelname' title='{html.escape(name)}'>{html.escape(name)}</div>"
        chips = []
        for met, val in metric_scores.items():
            if val is None:
                continue
            cls = score_class(val, metric_max.get(met, 5.0))
            chips.append(f"<span class='score {cls}'>"
                         f"{html.escape(met)}: {fmt_score(val)}</span>")
        if chips:
            out += f"<div class='scoreline'>{''.join(chips)}</div>"
        return out

    for i in indices:
        # Pull the first non-empty reference / instruction across models for this index.
        ref = ""
        instr = ""
        for plist in preds:
            if i < len(plist):
                p = plist[i]
                ref = ref or (p.get("reference") or "")
                instr = instr or (p.get("audio_text_instruction")
                                  or p.get("instruction") or "")
                if ref and instr:
                    break

        per_sample_scores = [
            s[i] if i < len(s) else None for s in all_scores
        ]
        head = f"#{i}"
        numeric = [v for v in per_sample_scores if isinstance(v, (int, float))]
        if len(numeric) >= 2:
            head += f" <span class='diff'>spread={max(numeric) - min(numeric):+.2f}</span>"

        parts.append("<div class='sample'>")
        parts.append(f"<h3>{head}</h3>")
        if instr:
            parts.append(f"<div class='instr'>{html.escape(instr)}</div>")
        parts.append("<div class='row'>")
        # Reference column mirrors the model columns' header (name line + scoreline) so
        # the text boxes align. Its scoreline shows each metric's spread (max−min) across
        # models at this sample.
        ref_chips = []
        for met in all_metrics:
            vals = [
                per_model_metric_scores[idx].get(met, [])[i]
                for idx in range(len(models))
                if i < len(per_model_metric_scores[idx].get(met, []))
                and isinstance(per_model_metric_scores[idx].get(met, [])[i], (int, float))
            ]
            if len(vals) >= 2:
                ref_chips.append(
                    f"<span class='score'>{html.escape(met)}: "
                    f"spread={max(vals) - min(vals):+.2f}</span>"
                )
        ref_scoreline = (
            f"<div class='scoreline'>{''.join(ref_chips)}</div>" if ref_chips else ""
        )
        parts.append(f"<div class='col'><div class='modelname'>Reference</div>{ref_scoreline}"
                     f"<div class='text ref'>{html.escape(str(ref))}</div></div>")
        for idx, m in enumerate(models):
            p = preds[idx][i] if i < len(preds[idx]) else {}
            cls = f"pred{(idx % len(PRED_PALETTE)) + 1}"
            metric_scores = {
                met: (scores[i] if i < len(scores) else None)
                for met, scores in per_model_metric_scores[idx].items()
            }
            parts.append(
                f"<div class='col'>{header_block(m, metric_scores)}"
                f"<div class='text {cls}'>"
                f"{html.escape(str(p.get('model_prediction','')))}</div></div>"
            )
        parts.append("</div></div>")

    parts.append("</body></html>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Results root (e.g. results, log_for_all_models)")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--dataset", "-d", help="Dataset name (filename without .json)")
    sel.add_argument("--task", "-t",
                     help="Task name (e.g. ASR, 'QUESTION ANSWERING'); generates one report "
                          "per dataset whose score file's `task` field matches (case-insensitive)")
    ap.add_argument("--models", "-m", nargs="+", required=True,
                    help="One or more model folder names")
    ap.add_argument("--language", help="Language subfolder (auto-detect if omitted)")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Output HTML path (default: comparisons/<dataset>_<lang>_<models>.html)")
    ap.add_argument("--output-dir", type=Path, default=Path("comparisons"),
                    help="Directory to write the report into (default: comparisons/)")
    ap.add_argument("--limit", type=int, default=0, help="Max samples to render (0=all)")
    ap.add_argument("--sort-by-diff", action="store_true",
                    help="Sort samples by score spread (max−min across models)")
    args = ap.parse_args()

    if args.task and args.output:
        print("[error] --output cannot be combined with --task (multiple files produced)",
              file=sys.stderr)
        sys.exit(2)

    if args.task:
        targets = discover_task_datasets(args.root, args.models[0], args.task, args.language)
        if not targets:
            print(f"[error] no datasets matching task={args.task!r} for model "
                  f"{args.models[0]}", file=sys.stderr)
            sys.exit(1)
        listed = ", ".join(f"{d}({l or '-'})" for d, l in targets)
        print(f"[task] {args.task}: {len(targets)} dataset(s) — {listed}")
        for ds, lang in targets:
            run_one(args, ds, lang)
    else:
        run_one(args, args.dataset, args.language)


def discover_task_datasets(root: Path, model: str, task: str, language: Optional[str]):
    """Scan <root>/<model>/[<lang>/]*_score.json and return list of (dataset, lang)
    whose `task` field matches `task` (case-insensitive)."""
    model_dir = root / model
    if not model_dir.is_dir():
        print(f"[error] model dir not found: {model_dir}", file=sys.stderr)
        return []

    if language:
        search_dirs = [(model_dir / language, language)]
    else:
        search_dirs = [(d, d.name) for d in model_dir.iterdir() if d.is_dir()]
        search_dirs.append((model_dir, None))

    target = task.strip().lower()
    found = []
    seen = set()
    for base, lang in search_dirs:
        if not base.is_dir():
            continue
        for score_path in base.glob("*_score.json"):
            data = load_json(score_path)
            if not isinstance(data, dict):
                continue
            t = (data.get("task") or "").strip().lower()
            if t != target:
                continue
            ds = score_path.name[: -len("_score.json")]
            key = (ds, lang)
            if key in seen:
                continue
            seen.add(key)
            found.append(key)
    return found


def run_one(args, dataset: str, language: Optional[str]):
    preds_list, scores_list, langs = [], [], []
    for m in args.models:
        p, s, lang = find_files(args.root, m, dataset, language)
        if p is None and s is None:
            print(f"[warn] skip {m}/{dataset}: no files found", file=sys.stderr)
            preds_list.append(None)
            scores_list.append(None)
            langs.append(None)
            continue
        preds_list.append(load_json(p))
        scores_list.append(load_json(s))
        langs.append(lang)

    if not any(p or s for p, s in zip(preds_list, scores_list)):
        print(f"[error] no files found for {dataset} across any model", file=sys.stderr)
        return

    resolved_lang = language or next((l for l in langs if l), None)

    for m, s in zip(args.models, scores_list):
        print_summary(m, s)

    if args.output is None:
        lang_part = f"_{resolved_lang}" if resolved_lang else ""
        models_part = "_vs_".join(args.models)
        if len(models_part) > 120:
            models_part = f"{args.models[0]}_vs_{len(args.models) - 1}_others"
        out = args.output_dir / f"{dataset}{lang_part}_{models_part}.html"
    else:
        out = args.output

    # Inject the per-call dataset/language so render_html sees them.
    render_args = argparse.Namespace(**vars(args))
    render_args.dataset = dataset
    render_args.language = resolved_lang

    out.parent.mkdir(parents=True, exist_ok=True)
    html_str = render_html(render_args, args.models, preds_list, scores_list)
    out.write_text(html_str)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
