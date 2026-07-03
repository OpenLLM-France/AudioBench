#!/usr/bin/env python3
"""Compare several models on a radar (spider) chart.

Reads the per-dataset ``*_score.json`` files written by
``audio_bench.main_evaluate`` (the ``results/`` layout, e.g.
``results/<model>/<LANG>/<dataset>_score.json``) and draws one polygon per
model over a set of axes (super-categories, tasks, or individual datasets).

Because the benchmark mixes metrics with different scales and directions
(WER lower-is-better, BLEU/judge-scores higher-is-better, accuracy 0-1, ...),
scores are normalized before plotting (each axis value is the mean of its
datasets' normalized scores):

* ``--normalize global`` (default) — absolute score against a fixed per-metric
  ideal (WER→0, BLEU/judge/acc→100), rescaled to ``[0, 1]``. Independent of the
  other models, so a tiny gap stays tiny and a large gap stays large. Honest
  when comparing only 2-3 models.
* ``--normalize minmax`` — rescale each dataset across the compared models so
  ``1.0 = best of them`` and ``0.0 = worst``. Good for spotting relative
  ordering, but exaggerates negligible gaps when few models are compared.
* ``--normalize raw`` — plot oriented raw scores (use with a single metric).

Outputs a PNG (matplotlib) and, unless ``--no-html``, an interactive HTML
radar (plotly).

Examples:
    # All models found under results/, axes = super-categories (ASR/AST/QA/...)
    python -m audio_bench.visualization.plot_radar results/

    # Pick models explicitly, one axis per fine-grained task
    python -m audio_bench.visualization.plot_radar results/ \
        -m phi_4_multimodal_instruct audio_flamingo_3 qwen2_audio_7b_instruct \
        --by task

    # One axis per dataset, restricted to French ASR, raw WER (inverted so
    # outward = better)
    python -m audio_bench.visualization.plot_radar results/ \
        --by dataset --language FR --normalize raw
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the score-loading / classification conventions from the HTML reporter so
# both tools agree on what a "score" is and which way is better.
from audio_bench.visualization.plot_results_to_html import (
    LOWER_IS_BETTER,
    ZERO_TO_ONE_RANGE,
    _SUPER_CATEGORY_ORDER,
    _TASK_METRIC_OVERRIDE,
    _display_score,
    _lang_sort_key,
    _model_color_map,
    _most_common_metric,
    _super_category,
    _task_display_name,
    load_all_scores,
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

# Subtitle shown under each figure title, per normalization mode.
_NORMALIZE_SUBTITLE = {
    "minmax": "min-max normalized per dataset across the compared models "
              "(outward = best of them; exaggerates ties)",
    "global": "absolute score vs a fixed per-metric ideal "
              "(WER→0, BLEU/judge/acc→100); outward = ideal",
    "raw": "raw scores, oriented so outward = better",
}


def _lang_suffix(entry):
    lang = entry.get("language")
    return f" [{lang}]" if lang and lang != "UNKNOWN" else ""


def _axis_of(entry, by):
    """Return the radar-axis label for an entry given the grouping mode."""
    if by == "super":
        return _super_category(entry.get("task") or "")
    if by == "super-language":
        return _super_category(entry.get("task") or "") + _lang_suffix(entry)
    if by == "task":
        # One axis per task, no folding. "Others" is reserved for --by super.
        return _task_display_name(entry.get("task") or "Unknown")
    if by == "task-language":
        return _task_display_name(entry.get("task") or "Unknown") + _lang_suffix(entry)
    if by == "dataset":
        return entry["dataset_name"] + _lang_suffix(entry)
    raise ValueError(f"unknown --by value: {by}")


def _dataset_key(entry):
    """Identity of a comparable cell: same dataset + language + metric."""
    return (entry["dataset_name"], entry.get("language"), entry["metric_name"])


def _split_axis(axis):
    """Split 'ASR [FR]' -> ('ASR', 'FR'); 'ASR' -> ('ASR', None)."""
    if axis.endswith("]") and " [" in axis:
        i = axis.rindex(" [")
        return axis[:i], axis[i + 2:-1]
    return axis, None


def _axis_sort_key(axis, by):
    """Order axes by super-category (canonical order), then group, then language,
    so related sectors sit next to each other instead of alphabetically."""
    group, lang = _split_axis(axis)
    if by in ("super", "super-language"):
        super_cat = group
    elif by in ("task", "task-language"):
        super_cat = _super_category(group.upper())
    else:  # dataset: no task info in the label
        super_cat = None
    cat_idx = (_SUPER_CATEGORY_ORDER.index(super_cat)
               if super_cat in _SUPER_CATEGORY_ORDER else len(_SUPER_CATEGORY_ORDER))
    return (cat_idx, group, _lang_sort_key(lang) if lang else ())


def _select_task_metrics(entries):
    """Keep a single metric per task, as report.html does.

    A dataset may be scored with several metrics (e.g. AST's Multilingual_TEDx
    has both ``bleu`` and ``meteor``). Counting both would double-weight that
    task on the radar and mix two scales, so we keep the task's override metric
    (``_TASK_METRIC_OVERRIDE``, e.g. AST→meteor) when present, else its most
    common metric.
    """
    by_task = defaultdict(list)
    for e in entries:
        by_task[(e.get("task") or "").upper()].append(e)
    chosen = {}
    for task, ents in by_task.items():
        override = _TASK_METRIC_OVERRIDE.get(task)
        if override and any(x["metric_name"] == override for x in ents):
            chosen[task] = override
        else:
            chosen[task] = _most_common_metric(ents)
    return [e for e in entries
            if e["metric_name"] == chosen[(e.get("task") or "").upper()]]


def _oriented_raw(entry):
    """Raw score oriented so that larger is better, on a readable scale.

    WER (and other lower-is-better metrics) are inverted; 0-1 metrics are
    scaled to 0-100 so the radius is comparable to BLEU/judge scores.
    """
    metric = entry["metric_name"]
    score = _display_score(entry["score"], metric)  # ×100 for 0-1 metrics
    if metric in LOWER_IS_BETTER:
        top = 100.0 if metric in ZERO_TO_ONE_RANGE else max(score, 100.0)
        return top - score
    return score


def build_axis_values(entries, models, by, normalize):
    """Return (axes, {model: {axis: value}}, axis_dataset_counts, hover).

    ``value`` is in [0, 1] for minmax/global, or the oriented raw score for raw
    mode. ``hover`` maps ``(model, axis) -> (mean_display_score, metric_label,
    n_datasets)`` for tooltips, where ``metric_label`` is the metric name when
    the axis is single-metric else ``"mixed"``.
    """
    # One metric per task (e.g. AST→meteor only), so a dataset scored with
    # several metrics is not counted/averaged twice.
    entries = _select_task_metrics(entries)

    # Group the comparable cells: cell_key -> {model: oriented_raw_score}
    cells = defaultdict(dict)
    cell_disp = defaultdict(dict)  # cell_key -> {model: display_score (raw, for hover)}
    cell_axis = {}
    cell_metric = {}
    for e in entries:
        if e["model_name"] not in models:
            continue
        key = _dataset_key(e)
        cells[key][e["model_name"]] = _oriented_raw(e)
        cell_disp[key][e["model_name"]] = _display_score(e["score"], e["metric_name"])
        cell_axis[key] = _axis_of(e, by)
        cell_metric[key] = e["metric_name"]

    # Keep only datasets that *every* compared model has, so no model can "win"
    # an axis just by being the only one evaluated on it. Dropped cells are
    # reported rather than silently neutralized.
    n_models = len(models)
    dropped = [(k, set(models) - set(ms.keys())) for k, ms in cells.items()
               if len(ms) < n_models]
    if dropped:
        print(f"[warn] {len(dropped)} dataset cell(s) dropped — not present for "
              f"all {n_models} compared models:", file=sys.stderr)
        for key, missing in sorted(dropped, key=lambda x: x[0])[:40]:
            ds, lang, metric = key
            label = f"{ds} [{lang}]" if lang and lang != "UNKNOWN" else ds
            print(f"         - {label} ({metric}) — missing for: "
                  f"{', '.join(sorted(missing))}", file=sys.stderr)
        if len(dropped) > 40:
            print(f"         ... and {len(dropped) - 40} more", file=sys.stderr)
    cells = {k: ms for k, ms in cells.items() if len(ms) == n_models}

    # Per cell, convert to the value we average over.
    # minmax: rescale across the participating models to [0, 1].
    per_model_axis_vals = defaultdict(lambda: defaultdict(list))
    per_model_axis_disp = defaultdict(lambda: defaultdict(list))  # for hover
    axis_metrics = defaultdict(set)
    axis_datasets = defaultdict(set)
    for key, model_scores in cells.items():
        axis = cell_axis[key]
        axis_datasets[axis].add(key)
        axis_metrics[axis].add(cell_metric[key])
        for m, dv in cell_disp[key].items():
            per_model_axis_disp[m][axis].append(dv)
        if normalize == "raw":
            for m, v in model_scores.items():
                per_model_axis_vals[m][axis].append(v)
            continue
        if normalize == "global":
            # _oriented_raw is already on a fixed 0-100 per-metric scale, so
            # divide by the ideal (100) and clamp. No dependence on the other
            # models -> a tiny gap stays tiny, a large gap stays large.
            for m, v in model_scores.items():
                per_model_axis_vals[m][axis].append(min(max(v / 100.0, 0.0), 1.0))
            continue
        lo = min(model_scores.values())
        hi = max(model_scores.values())
        span = hi - lo
        for m, v in model_scores.items():
            # All models are present here (others were dropped above); a zero
            # span now means a genuine tie -> neutral 0.5. Otherwise rescale.
            norm = 0.5 if span <= 1e-12 else (v - lo) / span
            per_model_axis_vals[m][axis].append(norm)

    axes = sorted(axis_datasets.keys(), key=lambda a: _axis_sort_key(a, by))
    model_axis_value = {}
    for m in models:
        model_axis_value[m] = {
            axis: float(np.mean(vals))
            for axis, vals in per_model_axis_vals.get(m, {}).items()
            if vals
        }
    axis_counts = {axis: len(ds) for axis, ds in axis_datasets.items()}
    axis_metric_label = {
        ax: (next(iter(ms)) if len(ms) == 1 else "mixed")
        for ax, ms in axis_metrics.items()
    }
    hover = {
        (m, ax): (float(np.mean(vals)), axis_metric_label[ax], axis_counts[ax])
        for m in models
        for ax, vals in per_model_axis_disp.get(m, {}).items()
        if vals
    }
    return axes, model_axis_value, axis_counts, hover


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _angles(n):
    return [i / n * 2 * math.pi for i in range(n)]


def render_png(axes, model_axis_value, axis_counts, colors, normalize, title, out_path):
    if len(axes) < 3:
        print(
            f"[warn] only {len(axes)} axis/axes ({', '.join(axes)}); a radar needs "
            "at least 3. Try a different --by/--language.",
            file=sys.stderr,
        )
    n = len(axes)
    angles = _angles(n)
    closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{a}\n({axis_counts.get(a, 0)} ds)" for a in axes], fontsize=9)

    if normalize in ("minmax", "global"):
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        outer = "best" if normalize == "minmax" else "ideal"
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", outer], fontsize=7)
    ax.tick_params(axis="y", labelcolor="#888")

    for model, axis_vals in model_axis_value.items():
        vals = [axis_vals.get(a, float("nan")) for a in axes]
        # Close the loop; matplotlib breaks lines on NaN automatically.
        line = vals + vals[:1]
        ax.plot(closed, line, color=colors[model], lw=2, label=model)
        ax.fill(closed, line, color=colors[model], alpha=0.08)

    sub = _NORMALIZE_SUBTITLE[normalize]
    ax.set_title(f"{title}\n{sub}", fontsize=13, pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def render_html(axes, model_axis_value, axis_counts, hover, colors, normalize, title, out_path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[warn] plotly not installed; skipping HTML output", file=sys.stderr)
        return
    theta = [f"{a} ({axis_counts.get(a, 0)} ds)" for a in axes]
    fig = go.Figure()
    for model, axis_vals in model_axis_value.items():
        r = [axis_vals.get(a) for a in axes]
        # customdata per point: [raw score string, normalized value] so the
        # tooltip shows the real metric value, not just the plotted radius.
        cdata = []
        for a in axes:
            info = hover.get((model, a))
            if info is None:
                cdata.append(["—"])
            else:
                raw, metric, _ = info
                cdata.append([f"{metric.upper()} {raw:.1f}"])
        fig.add_trace(go.Scatterpolar(
            r=r + r[:1],
            theta=theta + theta[:1],
            customdata=cdata + cdata[:1],
            name=model,
            line=dict(color=colors[model], width=2),
            fill="toself",
            opacity=0.7,
            connectgaps=False,
            hovertemplate=("<b>%{theta}</b><br>"
                           "score: %{customdata[0]}<br>"
                           "radius: %{r:.2f}"
                           "<extra>%{fullData.name}</extra>"),
        ))
    radial = dict(range=[0, 1]) if normalize in ("minmax", "global") else {}
    sub = _NORMALIZE_SUBTITLE[normalize]
    fig.update_layout(
        title=f"{title}<br><sup>{sub}</sup>",
        polar=dict(radialaxis=radial),
        legend=dict(font=dict(size=10)),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[ok] wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_folder", nargs="?", default="results",
                   help="folder with <model>/<LANG>/<dataset>_score.json (default: results)")
    p.add_argument("-m", "--models", nargs="*", default=None,
                   help="model_id (folder name) or corrected model_name to include; "
                        "default = all found")
    p.add_argument("--by",
                   choices=["super", "super-language", "task", "task-language", "dataset"],
                   default="super",
                   help="what each radar axis represents (default: super-category). "
                        "The *-language variants split every group per language "
                        "(e.g. 'ASR [FR]', 'ASR [EN]')")
    p.add_argument("--language", nargs="+", default=None, metavar="LANG",
                   help="keep only entries whose language is among these "
                        "(e.g. --language FR EN to restrict to French/English). "
                        "For translation pairs like FR-EN, every side must be in "
                        "the set. Entries with no/UNKNOWN language are dropped.")
    p.add_argument("--metric", default=None,
                   help="keep only entries computed with this metric (e.g. wer, bleu)")
    p.add_argument("--normalize", choices=["global", "minmax", "raw"], default="global",
                   help="global: absolute score vs a fixed per-metric ideal, [0,1] "
                        "(default, honest with few models); "
                        "minmax: rescale each dataset across the compared models to "
                        "[0,1] (exaggerates ties when comparing 2-3 models); "
                        "raw: plot oriented raw scores (use with a single metric)")
    p.add_argument("--show-all", action="store_true",
                   help="bypass the curated dataset/model filters in plot_results_to_html")
    p.add_argument("--output_folder", default="plots/",
                   help="where to write radar.png / radar.html (default: plots/, "
                        "same folder as plot_results_to_html's report.html)")
    p.add_argument("--title", default="AudioBench model comparison")
    p.add_argument("--no-html", action="store_true", help="skip the interactive HTML output")
    args = p.parse_args(argv)

    entries = load_all_scores(args.input_folder, show_all=args.show_all)
    if args.language:
        allowed = {l.upper() for l in args.language}
        def _lang_ok(e):
            lang = (e.get("language") or "").upper()
            if not lang or lang == "UNKNOWN":
                return False
            # Mono ("FR") or pair ("FR-EN"): every side must be allowed.
            return all(part in allowed for part in lang.split("-"))
        entries = [e for e in entries if _lang_ok(e)]
    if args.metric:
        entries = [e for e in entries if e["metric_name"] == args.metric]
    if not entries:
        print("[error] no score entries matched the given filters", file=sys.stderr)
        return 1

    all_models = sorted({e["model_name"] for e in entries})
    if args.models:
        # For each token: if it matches a model_name exactly, take only that
        # model; otherwise fall back to case-insensitive substring matching so
        # "flamingo" or "Phi-4" work without typing the full HF path. The
        # exact-first rule avoids a token like ".../Canary_Luciole-1B" also
        # pulling in ".../Canary_Luciole-1B_LLM-LoRA_8h".
        selected, missing = [], []
        for tok in args.models:
            if tok in all_models:
                hits = [tok]
            else:
                hits = [m for m in all_models if tok.lower() in m.lower()]
            if hits:
                selected.extend(hits)
            else:
                missing.append(tok)
        models = [m for m in all_models if m in set(selected)]
        if missing:
            print(f"[warn] no model matched: {', '.join(missing)}", file=sys.stderr)
            print(f"[info] available: {', '.join(all_models)}", file=sys.stderr)
        if not models:
            return 1
    else:
        models = all_models
    if len(models) < 2:
        print(f"[warn] only {len(models)} model(s); a comparison radar wants 2+.", file=sys.stderr)

    axes, model_axis_value, axis_counts, hover = build_axis_values(
        entries, models, args.by, args.normalize)
    if not axes:
        print("[error] no axes to plot after grouping", file=sys.stderr)
        return 1

    colors = _model_color_map(models)
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_png(axes, model_axis_value, axis_counts, colors, args.normalize,
               args.title, out_dir / "radar.png")
    if not args.no_html:
        render_html(axes, model_axis_value, axis_counts, hover, colors, args.normalize,
                    args.title, out_dir / "radar.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
