#!/usr/bin/env python3
"""Compare several models on a radar (spider) chart.

Reads the per-dataset ``*_score.json`` files written by
``audio_bench.main_evaluate`` (the ``results/`` layout, e.g.
``results/<model>/<LANG>/<dataset>_score.json``) and draws one polygon per
model over a set of axes (super-categories, tasks, or individual datasets).

Because the benchmark mixes metrics with different scales and directions
(WER lower-is-better, BLEU/judge-scores higher-is-better, accuracy 0-1, ...),
the default ``--normalize minmax`` mode rescales every dataset to ``[0, 1]``
across the compared models so that ``1.0 = best model on that dataset`` and
``0.0 = worst``. Each axis value is then the mean of its datasets' normalized
scores. Use ``--normalize raw`` only when every axis shares one metric.

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
from audio_bench.plot_results_to_html import (
    LOWER_IS_BETTER,
    ZERO_TO_ONE_RANGE,
    _display_score,
    _model_color_map,
    _super_category,
    _task_display_name,
    load_all_scores,
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _axis_of(entry, by):
    """Return the radar-axis label for an entry given the grouping mode."""
    if by == "super":
        return _super_category(entry.get("task") or "")
    if by == "task":
        return _task_display_name(entry.get("task") or "Unknown")
    if by == "dataset":
        name = entry["dataset_name"]
        lang = entry.get("language")
        return f"{name} [{lang}]" if lang and lang != "UNKNOWN" else name
    raise ValueError(f"unknown --by value: {by}")


def _dataset_key(entry):
    """Identity of a comparable cell: same dataset + language + metric."""
    return (entry["dataset_name"], entry.get("language"), entry["metric_name"])


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
    """Return (axes, {model: {axis: value}}, axis_dataset_counts).

    ``value`` is in [0, 1] for minmax mode, or the oriented raw score for raw
    mode. Models missing every dataset of an axis get no entry for that axis.
    """
    # Group the comparable cells: cell_key -> {model: oriented_raw_score}
    cells = defaultdict(dict)
    cell_axis = {}
    for e in entries:
        if e["model_name"] not in models:
            continue
        key = _dataset_key(e)
        cells[key][e["model_name"]] = _oriented_raw(e)
        cell_axis[key] = _axis_of(e, by)

    # Per cell, convert to the value we average over.
    # minmax: rescale across the participating models to [0, 1].
    per_model_axis_vals = defaultdict(lambda: defaultdict(list))
    axis_datasets = defaultdict(set)
    for key, model_scores in cells.items():
        axis = cell_axis[key]
        axis_datasets[axis].add(key)
        if normalize == "raw":
            for m, v in model_scores.items():
                per_model_axis_vals[m][axis].append(v)
            continue
        lo = min(model_scores.values())
        hi = max(model_scores.values())
        span = hi - lo
        for m, v in model_scores.items():
            # Degenerate cell (one model, or all tied) -> neutral 0.5 so it
            # neither rewards nor punishes; otherwise linear rescale.
            norm = 0.5 if span <= 1e-12 else (v - lo) / span
            per_model_axis_vals[m][axis].append(norm)

    axes = sorted(axis_datasets.keys())
    model_axis_value = {}
    for m in models:
        model_axis_value[m] = {
            axis: float(np.mean(vals))
            for axis, vals in per_model_axis_vals.get(m, {}).items()
            if vals
        }
    axis_counts = {axis: len(ds) for axis, ds in axis_datasets.items()}
    return axes, model_axis_value, axis_counts


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

    if normalize == "minmax":
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "best"], fontsize=7)
    ax.tick_params(axis="y", labelcolor="#888")

    for model, axis_vals in model_axis_value.items():
        vals = [axis_vals.get(a, float("nan")) for a in axes]
        # Close the loop; matplotlib breaks lines on NaN automatically.
        line = vals + vals[:1]
        ax.plot(closed, line, color=colors[model], lw=2, label=model)
        ax.fill(closed, line, color=colors[model], alpha=0.08)

    sub = "min-max normalized per dataset (outward = best)" if normalize == "minmax" \
        else "raw scores, oriented so outward = better"
    ax.set_title(f"{title}\n{sub}", fontsize=13, pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def render_html(axes, model_axis_value, axis_counts, colors, normalize, title, out_path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[warn] plotly not installed; skipping HTML output", file=sys.stderr)
        return
    theta = [f"{a} ({axis_counts.get(a, 0)} ds)" for a in axes]
    fig = go.Figure()
    for model, axis_vals in model_axis_value.items():
        r = [axis_vals.get(a) for a in axes]
        fig.add_trace(go.Scatterpolar(
            r=r + r[:1],
            theta=theta + theta[:1],
            name=model,
            line=dict(color=colors[model], width=2),
            fill="toself",
            opacity=0.7,
            connectgaps=False,
        ))
    radial = dict(range=[0, 1]) if normalize == "minmax" else {}
    sub = "min-max normalized per dataset (outward = best)" if normalize == "minmax" \
        else "raw scores, oriented so outward = better"
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
    p.add_argument("--by", choices=["super", "task", "dataset"], default="super",
                   help="what each radar axis represents (default: super-category)")
    p.add_argument("--language", default=None,
                   help="keep only entries for this language (e.g. FR, EN)")
    p.add_argument("--metric", default=None,
                   help="keep only entries computed with this metric (e.g. wer, bleu)")
    p.add_argument("--normalize", choices=["minmax", "raw"], default="minmax",
                   help="minmax: rescale each dataset across models to [0,1] (default); "
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
        entries = [e for e in entries if (e.get("language") or "").upper() == args.language.upper()]
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

    axes, model_axis_value, axis_counts = build_axis_values(
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
        render_html(axes, model_axis_value, axis_counts, colors, args.normalize,
                    args.title, out_dir / "radar.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
