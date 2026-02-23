#!/usr/bin/env python3
"""Plot AudioBench evaluation results as bar charts or image tables.

Usage examples:
    python src/plot_results.py results/
    python src/plot_results.py results/ --table
    python src/plot_results.py results/ --task asr
    python src/plot_results.py results/ --dataset librispeech_test_clean
    python src/plot_results.py results/ --aggregate --task asr
    python src/plot_results.py results/ --aggregate-by-language --task asr
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOWER_IS_BETTER = {"wer"}
ZERO_TO_ONE_RANGE = {"wer", "meteor"}

# For --aggregate mode: task -> list of dataset name prefixes
AGGREGATE_DATASETS = {
    "ASR": ["fleurs", "common_voice", "librispeech", "gigaspeech", "aishell",
            "earnings", "peoples_speech", "tedlium"],
    "ST":  ["covost2"],
    "SQA": ["slue_p2_sqa5", "spoken_squad", "public_sg_speech_qa",
            "cn_college_listen_mcq", "dream_tts_mcq"],
    "ASQA": ["clotho_aqa", "audiocaps_qa", "wavcaps_qa"],
    "AC":  ["audiocaps", "wavcaps"],
    "ER":  ["iemocap_emotion", "meld_sentiment", "meld_emotion"],
    "GR":  ["voxceleb_gender", "iemocap_gender"],
    "AR":  ["voxceleb_accent", "imda_ar"],
    "SI":  ["openhermes_audio", "alpaca_audio"],
    "SDS": ["imda_part3_30s_ds", "imda_part4_30s_ds",
            "imda_part5_30s_ds", "imda_part6_30s_ds"],
}

HIGHLIGHT_COLOR = "#b0c1d7"
MISSING_COLOR = "#e0e0e0"

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_all_scores(input_folder):
    """Scan input_folder/{model_dir}/**/*_score.json and return list of entry dicts.

    Supports the new results/ directory structure where score files may be nested
    in language subdirectories (e.g. results/model/FR/fleurs_score.json) and each
    file contains multiple metrics listed in data["metrics"].
    """
    entries = []
    input_path = Path(input_folder)

    for model_dir in sorted(input_path.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for filepath in sorted(model_dir.rglob("*_score.json")):
            dataset_name = filepath.name.removesuffix("_score.json")

            try:
                data = json.loads(filepath.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            metrics = data.get("metrics", [])
            if not metrics:
                continue

            task = data.get("task")
            language = data.get("language")

            for metric_name in metrics:
                try:
                    score = data[metric_name]
                except (KeyError, TypeError):
                    continue

                if not isinstance(score, (int, float)):
                    continue

                entries.append({
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "metric_name": metric_name,
                    "score": float(score),
                    "task": task,
                    "language": language,
                })

    return entries

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def normalize_task(task_str):
    """Normalize task string: 'ST-EN-ZH' -> 'ST', 'ASR-ZH' -> 'ASR'."""
    if task_str is None:
        return None
    parts = task_str.upper().split("-")
    return parts[0] if parts else None


def filter_entries(entries, task=None, dataset=None, language=None):
    """Filter entries by task, dataset, and/or language."""
    result = entries

    if dataset:
        result = [e for e in result if e["dataset_name"] == dataset]

    if task:
        task_upper = task.upper()
        result = [e for e in result
                  if e["task"] is not None and normalize_task(e["task"]) == task_upper]

    if language:
        lang_upper = language.upper()
        result = [e for e in result
                  if e["language"] is not None and e["language"].upper() == lang_upper]

    return result

# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def group_key(entry):
    """Return (task, language) grouping key for an entry."""
    t = entry["task"].upper() if entry["task"] else "UNKNOWN"
    l = entry["language"].upper() if entry["language"] else "UNKNOWN"
    return (t, l)


def group_entries_by_task_language(entries):
    """Group entries into {(task, language): [entries]} dict."""
    groups = defaultdict(list)
    for e in entries:
        groups[group_key(e)].append(e)
    return dict(groups)


def group_entries_by_metric(entries):
    """Group entries into {metric_name: [entries]} dict."""
    groups = defaultdict(list)
    for e in entries:
        groups[e["metric_name"]].append(e)
    return dict(groups)

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_entries(entries, task_filter=None, by_language=False):
    """Average scores across curated datasets per (model, task, metric).

    When *by_language* is True, datasets are first grouped by language within
    each task so that only datasets sharing the same language are averaged
    together (e.g. ASR_EN, ASR_ZH instead of a single ASR_MIXED).

    Returns new synthetic entries with dataset_name set to the task name
    (or task_language when *by_language* is True).
    """
    aggregated = []

    tasks_to_process = AGGREGATE_DATASETS
    if task_filter:
        tf = task_filter.upper()
        tasks_to_process = {k: v for k, v in AGGREGATE_DATASETS.items() if k == tf}

    all_models = sorted({e["model_name"] for e in entries})
    all_metrics = sorted({e["metric_name"] for e in entries})

    for agg_task, prefixes in tasks_to_process.items():
        # Collect matching entries for this task
        task_entries = [
            e for e in entries
            if any(e["dataset_name"].startswith(p) for p in prefixes)
        ]

        if by_language:
            # Sub-group by language
            lang_groups = defaultdict(list)
            for e in task_entries:
                lang = (e["language"] or "UNKNOWN").upper()
                lang_groups[lang].append(e)
        else:
            # Single group: all languages together
            lang_groups = {None: task_entries}

        for lang_key, lang_entries in sorted(lang_groups.items(), key=lambda x: x[0] or ""):
            for metric in all_metrics:
                for model in all_models:
                    scores = [
                        e["score"] for e in lang_entries
                        if e["model_name"] == model and e["metric_name"] == metric
                    ]

                    if not scores:
                        continue

                    if lang_key is not None:
                        # by_language mode: lang_key is the actual language
                        label = f"{agg_task}_{lang_key}_avg"
                        lang = lang_key
                    else:
                        # Original mode: detect language from entries
                        languages = {
                            (e["language"] or "UNKNOWN").upper()
                            for e in lang_entries
                            if e["model_name"] == model and e["metric_name"] == metric
                        }
                        lang = languages.pop() if len(languages) == 1 else "MIXED"
                        label = f"{agg_task}_avg"

                    aggregated.append({
                        "model_name": model,
                        "dataset_name": label,
                        "metric_name": metric,
                        "score": sum(scores) / len(scores),
                        "task": agg_task,
                        "language": lang,
                    })

    return aggregated

# ---------------------------------------------------------------------------
# Plotting — Bar Charts
# ---------------------------------------------------------------------------

def _model_color_map(models):
    """Assign a consistent color to each model name using tab20."""
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(models), 1))
    return {m: cmap(i) for i, m in enumerate(sorted(models))}


def _display_score(score, metric):
    """Format score for display: multiply by 100 for 0-1 range metrics."""
    if metric in ZERO_TO_ONE_RANGE:
        return score * 100
    return score


def _sort_ascending(metric):
    """Return True if lower is better for this metric."""
    return metric in LOWER_IS_BETTER


def plot_bar_charts(entries, title_prefix, output_folder):
    """Produce bar chart PNGs grouped by metric, one subplot per dataset."""
    by_metric = group_entries_by_metric(entries)
    all_models = sorted({e["model_name"] for e in entries})
    color_map = _model_color_map(all_models)
    os.makedirs(output_folder, exist_ok=True)

    for metric, metric_entries in sorted(by_metric.items()):
        # Organize: dataset -> model -> score
        ds_model_score = defaultdict(dict)
        for e in metric_entries:
            ds_model_score[e["dataset_name"]][e["model_name"]] = e["score"]

        datasets = sorted(ds_model_score.keys())
        if not datasets:
            continue

        ncols = min(3, len(datasets))
        nrows = (len(datasets) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 4 * nrows),
                                 squeeze=False)

        for idx, ds in enumerate(datasets):
            ax = axes[idx // ncols][idx % ncols]
            model_scores = ds_model_score[ds]

            ascending = _sort_ascending(metric)
            sorted_models = sorted(model_scores.keys(),
                                   key=lambda m: model_scores[m],
                                   reverse=not ascending)

            display_scores = [_display_score(model_scores[m], metric) for m in sorted_models]
            colors = [color_map[m] for m in sorted_models]

            bars = ax.bar(range(len(sorted_models)), display_scores, color=colors)

            # Value labels
            for bar, val in zip(bars, display_scores):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

            ax.set_xticks(range(len(sorted_models)))
            ax.set_xticklabels(sorted_models, rotation=45, ha="right", fontsize=7)
            ax.set_title(ds, fontsize=9)
            ax.set_ylabel(metric.upper() + (" (%)" if metric in ZERO_TO_ONE_RANGE else ""))

        # Hide unused subplots
        for idx in range(len(datasets), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"{title_prefix} — {metric.upper()}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        safe_prefix = title_prefix.replace("/", "_").replace(" ", "_")
        out_path = Path(output_folder) / f"{safe_prefix}_{metric}_bar.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# Plotting — Table
# ---------------------------------------------------------------------------

def plot_table(entries, title_prefix, output_folder):
    """Render a comparison table as a PNG image, one per metric."""
    by_metric = group_entries_by_metric(entries)
    os.makedirs(output_folder, exist_ok=True)

    for metric, metric_entries in sorted(by_metric.items()):
        # Build dataset -> model -> score
        ds_model_score = defaultdict(dict)
        for e in metric_entries:
            ds_model_score[e["dataset_name"]][e["model_name"]] = e["score"]

        datasets = sorted(ds_model_score.keys())
        all_models = sorted({e["model_name"] for e in metric_entries})
        if not datasets or not all_models:
            continue

        ascending = _sort_ascending(metric)

        # Compute average per model (over datasets where score exists)
        model_avg = {}
        for m in all_models:
            scores = [ds_model_score[ds][m] for ds in datasets if m in ds_model_score[ds]]
            model_avg[m] = sum(scores) / len(scores) if scores else None

        # Sort models: best average first
        sorted_models = sorted(
            all_models,
            key=lambda m: (model_avg[m] is None, model_avg[m] if model_avg[m] is not None else 0),
            reverse=not ascending,
        )

        col_labels = datasets + ["Average"]
        row_labels = sorted_models

        # Build cell text and find best per column
        cell_text = []
        for m in sorted_models:
            row = []
            for ds in datasets:
                if m in ds_model_score[ds]:
                    row.append(f"{_display_score(ds_model_score[ds][m], metric):.2f}")
                else:
                    row.append("-")
            avg = model_avg[m]
            row.append(f"{_display_score(avg, metric):.2f}" if avg is not None else "-")
            cell_text.append(row)

        # Find best score per column
        best_per_col = []
        for ci in range(len(col_labels)):
            vals = []
            for ri, m in enumerate(sorted_models):
                txt = cell_text[ri][ci]
                if txt != "-":
                    vals.append((float(txt), ri))
            if vals:
                if ascending:
                    best_per_col.append(min(vals, key=lambda x: x[0])[1])
                else:
                    best_per_col.append(max(vals, key=lambda x: x[0])[1])
            else:
                best_per_col.append(None)

        # Draw table
        fig_width = max(8, 1.5 * len(col_labels) + 2)
        fig_height = max(3, 0.4 * len(sorted_models) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)

        # Color cells
        for ri in range(len(sorted_models)):
            for ci in range(len(col_labels)):
                cell = table[ri + 1, ci]  # +1 because row 0 is header
                if cell_text[ri][ci] == "-":
                    cell.set_facecolor(MISSING_COLOR)
                elif best_per_col[ci] == ri:
                    cell.set_facecolor(HIGHLIGHT_COLOR)

        unit = " (%)" if metric in ZERO_TO_ONE_RANGE else ""
        fig.suptitle(f"{title_prefix} — {metric.upper()}{unit}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        safe_prefix = title_prefix.replace("/", "_").replace(" ", "_")
        out_path = Path(output_folder) / f"{safe_prefix}_{metric}_table.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot AudioBench evaluation results as bar charts or tables."
    )
    parser.add_argument("input_folder", help="Path to results folder (e.g. results/)")
    parser.add_argument("--table", action="store_true", help="Render image table instead of bar charts")
    parser.add_argument("--task", type=str, default=None, help="Filter by task (e.g. asr, st)")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by specific dataset name")
    parser.add_argument("--language", type=str, default=None, help="Filter by language (e.g. fr)")
    parser.add_argument("--output_folder", type=str, default="plots/", help="Where to save PNG images")
    parser.add_argument("--aggregate", action="store_true",
                        help="Average scores across curated datasets per task")
    parser.add_argument("--aggregate-by-language", action="store_true",
                        help="Like --aggregate, but average only datasets sharing the same language")
    args = parser.parse_args()

    # Load all scores
    entries = load_all_scores(args.input_folder)
    if not entries:
        print(f"No score files found in {args.input_folder}")
        return

    print(f"Loaded {len(entries)} score entries from {len({e['model_name'] for e in entries})} models")

    # Apply filters
    entries = filter_entries(entries, task=args.task, dataset=args.dataset, language=args.language)
    if not entries:
        print("No entries after filtering.")
        return

    print(f"{len(entries)} entries after filtering")

    plot_fn = plot_table if args.table else plot_bar_charts

    if args.dataset:
        # Single dataset mode
        title = args.dataset
        plot_fn(entries, title, args.output_folder)

    elif args.aggregate or args.aggregate_by_language:
        # Aggregate mode
        agg_entries = aggregate_entries(
            entries, task_filter=args.task, by_language=args.aggregate_by_language,
        )
        if not agg_entries:
            print("No aggregated entries produced.")
            return
        groups = group_entries_by_task_language(agg_entries)
        for (task, lang), group_entries in sorted(groups.items()):
            title = f"{task}_{lang}"
            plot_fn(group_entries, title, args.output_folder)

    else:
        # Group by (task, language), entries without metadata go to UNKNOWN
        groups = group_entries_by_task_language(entries)
        for (task, lang), group_entries in sorted(groups.items()):
            title = f"{task}_{lang}"
            plot_fn(group_entries, title, args.output_folder)


if __name__ == "__main__":
    main()
