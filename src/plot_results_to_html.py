#!/usr/bin/env python3
"""Plot AudioBench evaluation results as a single interactive HTML report.

Generates one HTML file with an Overview table (all tasks × models ranked)
and per-task Summary sections (violin plots + language-column tables with
expandable per-dataset sub-columns).

Output: {output_folder}/report.html

Usage examples:
    python src/plot_results_to_html.py results/
    python src/plot_results_to_html.py results/ --output_folder my_plots/
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
import plotly.express.colors as pxcolors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOWER_IS_BETTER = {"wer"}
ZERO_TO_ONE_RANGE = {"wer", "meteor"}

# Task -> list of dataset name prefixes for aggregation
AGGREGATE_DATASETS = {
    "ASR": ["fleurs", "common_voice", "librispeech", "gigaspeech", "aishell",
            "earnings", "peoples_speech", "tedlium"],
    "AST":  ["covost2"],
    "Question Answering": ["slue_p2_sqa5", "spoken_squad", "public_sg_speech_qa",
            "cn_college_listen_mcq", "dream_tts_mcq"],
    "Audio Question Answering": ["clotho_aqa", "audiocaps_qa", "wavcaps_qa"],
    "Audio Captioning":  ["audiocaps", "wavcaps"],
    "Emotion Recognition":  ["iemocap_emotion", "meld_sentiment", "meld_emotion"],
    "Gender Recognition":  ["voxceleb_gender", "iemocap_gender"],
    "Accent Recognition":  ["voxceleb_accent", "imda_ar"],
    "Instruction Following":  ["openhermes_audio", "alpaca_audio"],
    "Dialogue Summarization": ["imda_part3_30s_ds", "imda_part4_30s_ds",
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
            sub_task = data.get("sub_task")

            for metric_name in metrics:
                try:
                    score = data[metric_name]
                except (KeyError, TypeError):
                    continue

                # Judge metrics store scores as dicts with a "judge_score" key
                if isinstance(score, dict):
                    score = score.get("judge_score")
                    if score is None:
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
                    "sub_task": sub_task,
                })

    return entries

# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def group_entries_by_metric(entries):
    """Group entries into {metric_name: [entries]} dict."""
    groups = defaultdict(list)
    for e in entries:
        groups[e["metric_name"]].append(e)
    return dict(groups)

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_entries(entries, task_filter=None, by_language=False, by_subtask=False):
    """Average scores across curated datasets per (model, task, metric).

    When *by_language* is True, datasets are first grouped by language within
    each task so that only datasets sharing the same language are averaged
    together (e.g. ASR_EN, ASR_ZH instead of a single ASR_MIXED).

    When *by_subtask* is True, datasets are grouped by effective sub-task
    (sub_task field for non-ASR/AST tasks, language otherwise).

    *by_language* and *by_subtask* are mutually exclusive.

    Returns new synthetic entries with dataset_name set to the task name
    (or task_language / sub_task when grouping is active).
    """
    aggregated = []

    tasks_to_process = AGGREGATE_DATASETS
    if task_filter:
        tf = task_filter.upper()
        tasks_to_process = {k: v for k, v in AGGREGATE_DATASETS.items() if k.upper() == tf}

    all_models = sorted({e["model_name"] for e in entries})
    all_metrics = sorted({e["metric_name"] for e in entries})

    for agg_task, prefixes in tasks_to_process.items():
        # Collect matching entries for this task
        task_entries = [
            e for e in entries
            if any(e["dataset_name"].startswith(p) for p in prefixes)
        ]

        if by_subtask:
            # Sub-group by effective sub-task
            lang_groups = defaultdict(list)
            for e in task_entries:
                lang_groups[_effective_subtask(e)].append(e)
        elif by_language:
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
                        # by_language mode: use language as subplot title
                        label = lang_key
                        lang = lang_key
                    else:
                        # aggregate mode: use task as subplot title
                        languages = {
                            (e["language"] or "UNKNOWN").upper()
                            for e in lang_entries
                            if e["model_name"] == model and e["metric_name"] == metric
                        }
                        lang = languages.pop() if len(languages) == 1 else "MIXED"
                        label = agg_task

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
# Shared plot helpers
# ---------------------------------------------------------------------------

_PLOTLY_PALETTE = pxcolors.qualitative.D3

def _model_color_map(models):
    """Assign a consistent color to each model name using the Plotly D3 palette."""
    palette = _PLOTLY_PALETTE
    return {m: palette[i % len(palette)] for i, m in enumerate(sorted(models))}


def _display_score(score, metric):
    """Format score for display: multiply by 100 for 0-1 range metrics."""
    if metric in ZERO_TO_ONE_RANGE:
        return score * 100
    return score


def _sort_ascending(metric):
    """Return True if lower is better for this metric."""
    return metric in LOWER_IS_BETTER


def _effective_subtask(entry):
    """Grouping key for sub-task expansion.

    ASR/AST -> language; others -> sub_task if set, else language.
    """
    task = (entry.get("task") or "").upper()
    if task in ("ASR", "AST"):
        return (entry.get("language") or "UNKNOWN").upper()
    st = entry.get("sub_task")
    if st:
        return st
    return (entry.get("language") or "UNKNOWN").upper()


def _prepare_metric_data(entries):
    """Yield (metric, ds_model_score, datasets, models) per metric."""
    by_metric = group_entries_by_metric(entries)
    for metric, metric_entries in sorted(by_metric.items()):
        ds_model_score = defaultdict(dict)
        for e in metric_entries:
            ds_model_score[e["dataset_name"]][e["model_name"]] = e["score"]
        datasets = sorted(ds_model_score.keys())
        models = sorted({e["model_name"] for e in metric_entries})
        if datasets and models:
            yield metric, ds_model_score, datasets, models


def _format_suptitle(title_prefix, metric):
    """Produce a consistent, unit-aware suptitle."""
    unit = " (%)" if metric in ZERO_TO_ONE_RANGE else ""
    return f"{title_prefix} — {metric.upper()}{unit}"


# ---------------------------------------------------------------------------
# Plotting — Violin Charts
# ---------------------------------------------------------------------------

def plot_violin_charts(entries, title_prefix, collector):
    """Produce violin plots showing per-dataset score distributions per model.

    Each metric gets one figure with one violin trace per model.  Individual
    dataset scores are overlaid as jittered points with hover text.
    """
    all_models = sorted({e["model_name"] for e in entries})
    color_map = _model_color_map(all_models)

    for metric, ds_model_score, datasets, models in _prepare_metric_data(entries):
        ascending = _sort_ascending(metric)

        # Sort models by average score (best first)
        model_avg = {}
        for m in models:
            scores = [ds_model_score[ds][m] for ds in datasets if m in ds_model_score[ds]]
            model_avg[m] = sum(scores) / len(scores) if scores else None

        sorted_models = sorted(
            models,
            key=lambda m: (model_avg[m] is None,
                           model_avg[m] if model_avg[m] is not None else 0),
            reverse=not ascending,
        )

        fig = go.Figure()

        for m in sorted_models:
            scores_raw = []
            hover_texts = []
            for ds in datasets:
                if m in ds_model_score[ds]:
                    scores_raw.append(ds_model_score[ds][m])
                    hover_texts.append(
                        f"{ds}: {_display_score(ds_model_score[ds][m], metric):.2f}"
                    )

            display_vals = [_display_score(s, metric) for s in scores_raw]

            fig.add_trace(go.Violin(
                y=display_vals,
                name=m,
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.3,
                pointpos=0,
                marker=dict(color=color_map[m], size=5, opacity=0.7),
                line=dict(color=color_map[m]),
                fillcolor=color_map[m],
                opacity=0.5,
                hovertext=hover_texts,
                hoverinfo="text+name",
                showlegend=False,
            ))

        unit = " (%)" if metric in ZERO_TO_ONE_RANGE else ""
        fig.update_layout(
            title_text=_format_suptitle(title_prefix, metric),
            title_font_size=16,
            yaxis_title=f"{metric.upper()}{unit}",
            height=450,
            width=max(600, 120 * len(sorted_models)),
            template="plotly_white",
            xaxis_tickangle=45,
            xaxis_tickfont_size=10,
        )

        collector.append({
            "category": title_prefix,
            "chart_type": "violin",
            "metric": metric,
            "fig": fig,
        })

# ---------------------------------------------------------------------------
# Plotting — Overview Table (all tasks × models)
# ---------------------------------------------------------------------------

def plot_overview_table(entries, collector):
    """Build an overview HTML table with expandable per-language sub-columns.

    Each task column has a [+] toggle button that reveals hidden per-language
    sub-columns via client-side JavaScript.  Tasks with only one language
    have no toggle (expanding would just duplicate the aggregate value).
    """
    agg = aggregate_entries(entries, by_language=False)
    if not agg:
        return

    agg_lang = aggregate_entries(entries, by_language=True)
    agg_sub = aggregate_entries(entries, by_subtask=True)

    # Group by task
    task_entries = defaultdict(list)
    for e in agg:
        task_entries[e["task"]].append(e)

    # For each task pick the most common metric
    task_metric = {}
    for task, ents in task_entries.items():
        metric_counts = defaultdict(int)
        for e in ents:
            metric_counts[e["metric_name"]] += 1
        task_metric[task] = max(metric_counts, key=metric_counts.get)

    tasks = sorted(task_metric.keys())
    if not tasks:
        return

    all_models = sorted({e["model_name"] for e in agg})

    # Build score lookup: task -> model -> score
    task_model_score = {}
    for task in tasks:
        metric = task_metric[task]
        model_scores = {}
        for m in all_models:
            scores = [
                e["score"] for e in task_entries[task]
                if e["model_name"] == m and e["metric_name"] == metric
            ]
            if scores:
                model_scores[m] = sum(scores) / len(scores)
        task_model_score[task] = model_scores

    # Build per-language scores: task -> lang -> model -> score
    task_lang_scores = defaultdict(lambda: defaultdict(dict))
    for e in agg_lang:
        task = e["task"]
        if task not in task_metric:
            continue
        if e["metric_name"] != task_metric[task]:
            continue
        lang = (e["language"] or "UNKNOWN").upper()
        task_lang_scores[task][lang][e["model_name"]] = e["score"]

    task_languages = {t: sorted(task_lang_scores[t].keys()) for t in tasks}
    expandable_tasks = {t for t in tasks if len(task_languages.get(t, [])) >= 2}

    # Build per-subtask scores: task -> subtask_key -> model -> score
    task_sub_scores = defaultdict(lambda: defaultdict(dict))
    for e in agg_sub:
        task = e["task"]
        if task not in task_metric:
            continue
        if e["metric_name"] != task_metric[task]:
            continue
        sub_key = (e["language"] or "UNKNOWN").upper()  # agg uses language field for label
        task_sub_scores[task][sub_key][e["model_name"]] = e["score"]

    task_subtasks = {t: sorted(task_sub_scores[t].keys()) for t in tasks}

    # dual_tasks: tasks where language and sub_task groupings differ and both have >=2 groups
    dual_tasks = set()
    for t in tasks:
        langs = task_languages.get(t, [])
        subs = task_subtasks.get(t, [])
        if len(langs) >= 2 and len(subs) >= 2 and langs != subs:
            dual_tasks.add(t)

    # Compute per-task ranks (1 = best)
    task_model_rank = {}
    for task in tasks:
        ascending = _sort_ascending(task_metric[task])
        scores = task_model_score[task]
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)
        task_model_rank[task] = {m: rank + 1 for rank, (m, _) in enumerate(ranked)}

    # Compute average rank per model
    model_avg_rank = {}
    for m in all_models:
        ranks = [task_model_rank[t][m] for t in tasks if m in task_model_rank[t]]
        model_avg_rank[m] = sum(ranks) / len(ranks) if ranks else float("inf")

    sorted_models = sorted(all_models, key=lambda m: model_avg_rank[m])

    # --- Determine best-value row indices for highlighting ---
    def _best_row(vals_ri_pairs, ascending):
        if not vals_ri_pairs:
            return None
        return (min if ascending else max)(vals_ri_pairs, key=lambda x: x[0])[1]

    task_best = {}
    for task in tasks:
        metric = task_metric[task]
        asc = _sort_ascending(metric)
        pairs = [
            (_display_score(task_model_score[task][m], metric), ri)
            for ri, m in enumerate(sorted_models) if m in task_model_score[task]
        ]
        task_best[task] = _best_row(pairs, asc)

    task_lang_best = defaultdict(dict)
    for task in expandable_tasks | dual_tasks:
        metric = task_metric[task]
        asc = _sort_ascending(metric)
        for lang in task_languages[task]:
            pairs = [
                (_display_score(task_lang_scores[task][lang][m], metric), ri)
                for ri, m in enumerate(sorted_models) if m in task_lang_scores[task][lang]
            ]
            task_lang_best[task][lang] = _best_row(pairs, asc)

    task_sub_best = defaultdict(dict)
    for task in dual_tasks:
        metric = task_metric[task]
        asc = _sort_ascending(metric)
        for sub in task_subtasks[task]:
            pairs = [
                (_display_score(task_sub_scores[task][sub][m], metric), ri)
                for ri, m in enumerate(sorted_models) if m in task_sub_scores[task][sub]
            ]
            task_sub_best[task][sub] = _best_row(pairs, asc)

    rank_pairs = [
        (model_avg_rank[m], ri)
        for ri, m in enumerate(sorted_models) if model_avg_rank[m] != float("inf")
    ]
    rank_best = _best_row(rank_pairs, ascending=True)

    # --- Build HTML ---
    def _task_slug(task):
        return re.sub(r'[^a-zA-Z0-9]+', '_', task).strip('_')

    def _td(value_str, is_best=False, is_missing=False, extra_attrs=""):
        if is_missing:
            style = f' style="background:{MISSING_COLOR}"'
        elif is_best:
            style = f' style="background:{HIGHLIGHT_COLOR}"'
        else:
            style = ""
        return f"<td{extra_attrs}{style}>{value_str}</td>"

    lines = []

    # Scoped CSS
    lines.append(f"""\
<style>
.ov-tbl {{ border-collapse: collapse; font-family: inherit; font-size: 11px; margin: 8px 0; }}
.ov-tbl th, .ov-tbl td {{ padding: 7px 10px; border: 1px solid #e2e8f0; text-align: center; white-space: nowrap; }}
.ov-tbl thead th {{ background: #3a5a8c; color: white; font-weight: 600; }}
.ov-tbl tbody td:first-child {{ text-align: left; font-weight: 500; }}
.ov-tbl .lang-col {{ display: none; }}
.ov-tbl thead th a {{ color: white; text-decoration: none; border-bottom: 1px dashed rgba(255,255,255,.45); }}
.ov-tbl thead th a:hover {{ border-bottom-style: solid; }}
.ov-tbl .toggle-btn {{ cursor: pointer; margin-left: 4px; font-size: 9px;
  background: rgba(255,255,255,.25); border: 1px solid rgba(255,255,255,.4);
  color: white; border-radius: 3px; padding: 1px 5px; vertical-align: middle; }}
.ov-tbl .toggle-btn:hover {{ background: rgba(255,255,255,.45); }}
</style>""")

    lines.append('<table class="ov-tbl" id="overview-tbl">')

    # --- Header ---
    lines.append("<thead><tr>")
    lines.append("<th>Model</th>")
    for task in tasks:
        metric = task_metric[task]
        unit = " %" if metric in ZERO_TO_ONE_RANGE else ""
        summary_cat = "Summary \u00b7 " + task
        section_anchor = f"cat-{_slug(summary_cat)}"
        label = f'<a href="#{section_anchor}">{task}</a> ({metric.upper()}{unit})'
        slug = _task_slug(task)
        if task in dual_tasks:
            # Two buttons: [L+] for language, [S+] for sub-task
            lines.append(
                f'<th>{label} '
                f'<button class="toggle-btn toggle-lang-{slug}" '
                f'onclick="toggleOvGroup(this,\'{slug}\',\'lang\')">L+</button>'
                f'<button class="toggle-btn toggle-sub-{slug}" '
                f'onclick="toggleOvGroup(this,\'{slug}\',\'sub\')">S+</button></th>'
            )
            for lang in task_languages[task]:
                lines.append(f'<th class="lang-col" data-task-lang="{slug}">{lang}</th>')
            for sub in task_subtasks[task]:
                lines.append(f'<th class="lang-col" data-task-sub="{slug}">{sub}</th>')
        elif task in expandable_tasks:
            lines.append(
                f'<th>{label} '
                f'<button class="toggle-btn" onclick="toggleOvLang(this,\'{slug}\')">+</button></th>'
            )
            for lang in task_languages[task]:
                lines.append(f'<th class="lang-col" data-task="{slug}">{lang}</th>')
        else:
            lines.append(f"<th>{label}</th>")
    lines.append("<th>Avg Rank</th>")
    lines.append("</tr></thead>")

    # --- Body ---
    lines.append("<tbody>")
    for ri, m in enumerate(sorted_models):
        lines.append("<tr>")
        lines.append(f"<td>{m}</td>")

        for task in tasks:
            metric = task_metric[task]
            slug = _task_slug(task)

            # Aggregate cell
            if m in task_model_score[task]:
                val = f"{_display_score(task_model_score[task][m], metric):.2f}"
                lines.append(_td(val, is_best=(ri == task_best.get(task))))
            else:
                lines.append(_td("-", is_missing=True))

            # Per-language / per-subtask sub-cells (hidden by default)
            if task in dual_tasks:
                for lang in task_languages[task]:
                    attr = f' class="lang-col" data-task-lang="{slug}"'
                    if m in task_lang_scores[task][lang]:
                        val = f"{_display_score(task_lang_scores[task][lang][m], metric):.2f}"
                        lines.append(_td(val, is_best=(ri == task_lang_best[task].get(lang)),
                                         extra_attrs=attr))
                    else:
                        lines.append(_td("-", is_missing=True, extra_attrs=attr))
                for sub in task_subtasks[task]:
                    attr = f' class="lang-col" data-task-sub="{slug}"'
                    if m in task_sub_scores[task][sub]:
                        val = f"{_display_score(task_sub_scores[task][sub][m], metric):.2f}"
                        lines.append(_td(val, is_best=(ri == task_sub_best[task].get(sub)),
                                         extra_attrs=attr))
                    else:
                        lines.append(_td("-", is_missing=True, extra_attrs=attr))
            elif task in expandable_tasks:
                for lang in task_languages[task]:
                    attr = f' class="lang-col" data-task="{slug}"'
                    if m in task_lang_scores[task][lang]:
                        val = f"{_display_score(task_lang_scores[task][lang][m], metric):.2f}"
                        lines.append(_td(val, is_best=(ri == task_lang_best[task].get(lang)),
                                         extra_attrs=attr))
                    else:
                        lines.append(_td("-", is_missing=True, extra_attrs=attr))

        # Avg Rank
        r = model_avg_rank[m]
        if r != float("inf"):
            lines.append(_td(f"{r:.1f}", is_best=(ri == rank_best)))
        else:
            lines.append(_td("-", is_missing=True))

        lines.append("</tr>")

    lines.append("</tbody></table>")

    # JavaScript toggle
    lines.append("""\
<script>
function toggleOvLang(btn, task) {
  var tbl = document.getElementById('overview-tbl');
  var cells = tbl.querySelectorAll('[data-task="' + task + '"]');
  if (!cells.length) return;
  var show = cells[0].style.display !== 'table-cell';
  for (var i = 0; i < cells.length; i++)
    cells[i].style.display = show ? 'table-cell' : 'none';
  btn.textContent = show ? '\\u2212' : '+';
}
function toggleOvGroup(btn, task, group) {
  var tbl = document.getElementById('overview-tbl');
  var other = group === 'lang' ? 'sub' : 'lang';
  var cells = tbl.querySelectorAll('[data-task-' + group + '="' + task + '"]');
  var otherCells = tbl.querySelectorAll('[data-task-' + other + '="' + task + '"]');
  if (!cells.length) return;
  var show = cells[0].style.display !== 'table-cell';
  // Hide the other group first
  for (var i = 0; i < otherCells.length; i++)
    otherCells[i].style.display = 'none';
  // Find and reset the other button
  var otherBtn = tbl.querySelector('.toggle-' + other + '-' + task);
  if (otherBtn) otherBtn.textContent = other === 'lang' ? 'L+' : 'S+';
  // Toggle this group
  for (var i = 0; i < cells.length; i++)
    cells[i].style.display = show ? 'table-cell' : 'none';
  btn.textContent = show ? (group === 'lang' ? 'L\\u2212' : 'S\\u2212')
                         : (group === 'lang' ? 'L+' : 'S+');
}
</script>""")

    collector.append({
        "category": "Overview",
        "chart_type": "table",
        "metric": "overview",
        "raw_html": "\n".join(lines),
    })


# ---------------------------------------------------------------------------
# Plotting — Summary Tables (per-task, expandable per-dataset)
# ---------------------------------------------------------------------------

def plot_summary_tables(raw_entries, collector):
    """Build summary HTML tables for all tasks with expandable per-dataset sub-columns.

    Each language column has a [+] toggle that reveals the individual dataset
    scores, and a clickable link to the corresponding per-dataset section.

    When sub_task grouping differs from language grouping, both views are
    rendered with a toggle bar above the tables.
    """
    for task, prefixes in sorted(AGGREGATE_DATASETS.items()):
        task_raw = [
            e for e in raw_entries
            if any(e["dataset_name"].startswith(p) for p in prefixes)
        ]
        if not task_raw:
            continue

        agg_lang = aggregate_entries(raw_entries, task_filter=task, by_language=True)
        if not agg_lang:
            continue

        agg_sub = aggregate_entries(raw_entries, task_filter=task, by_subtask=True)

        # Check if sub-task grouping differs from language grouping
        lang_keys = sorted({e["dataset_name"] for e in agg_lang})
        sub_keys = sorted({e["dataset_name"] for e in agg_sub})
        has_dual = len(lang_keys) >= 2 and len(sub_keys) >= 2 and lang_keys != sub_keys

        for metric in sorted({e["metric_name"] for e in agg_lang}):
            if has_dual:
                _build_dual_summary_tables(task, metric, task_raw,
                                           agg_lang, agg_sub, collector)
            else:
                _build_summary_table(task, metric, task_raw, agg_lang,
                                     collector, group_key_fn=None)


def _build_summary_table(task, metric, task_raw, agg_lang, collector,
                         group_key_fn=None, tbl_id_suffix=""):
    """Build one HTML summary table for a (task, metric) pair.

    *group_key_fn*, when provided, maps a raw entry to its group label
    (used for sub-task grouping).  When None, groups by language.
    *tbl_id_suffix* is appended to the table HTML id for uniqueness.
    """
    agg_metric = [e for e in agg_lang if e["metric_name"] == metric]
    raw_metric = [e for e in task_raw if e["metric_name"] == metric]
    if not agg_metric:
        return

    all_models = sorted({e["model_name"] for e in agg_metric})
    languages = sorted({e["dataset_name"] for e in agg_metric})  # dataset_name = group key

    # lang -> model -> aggregated score
    lang_model_score = defaultdict(dict)
    for e in agg_metric:
        lang_model_score[e["dataset_name"]][e["model_name"]] = e["score"]

    # Per-dataset breakdown: group -> dataset -> model -> score
    lang_ds_model = defaultdict(lambda: defaultdict(dict))
    lang_datasets = defaultdict(set)
    for e in raw_metric:
        lang = group_key_fn(e) if group_key_fn else (e["language"] or "UNKNOWN").upper()
        lang_ds_model[lang][e["dataset_name"]][e["model_name"]] = e["score"]
        lang_datasets[lang].add(e["dataset_name"])
    lang_datasets = {l: sorted(ds) for l, ds in lang_datasets.items()}

    expandable_langs = {l for l in languages if len(lang_datasets.get(l, [])) >= 2}

    # Average per model across languages
    ascending = _sort_ascending(metric)
    model_avg = {}
    for m in all_models:
        scores = [lang_model_score[l][m] for l in languages if m in lang_model_score[l]]
        model_avg[m] = sum(scores) / len(scores) if scores else None

    sorted_models = sorted(
        all_models,
        key=lambda m: (model_avg[m] is None, model_avg[m] if model_avg[m] is not None else 0),
        reverse=not ascending,
    )

    # --- Best-value row indices for highlighting ---
    def _best_row(pairs, asc):
        if not pairs:
            return None
        return (min if asc else max)(pairs, key=lambda x: x[0])[1]

    lang_best = {}
    for lang in languages:
        pairs = [
            (_display_score(lang_model_score[lang][m], metric), ri)
            for ri, m in enumerate(sorted_models) if m in lang_model_score[lang]
        ]
        lang_best[lang] = _best_row(pairs, ascending)

    lang_ds_best = defaultdict(dict)
    for lang in expandable_langs:
        for ds in lang_datasets.get(lang, []):
            pairs = [
                (_display_score(lang_ds_model[lang][ds][m], metric), ri)
                for ri, m in enumerate(sorted_models) if m in lang_ds_model[lang][ds]
            ]
            lang_ds_best[lang][ds] = _best_row(pairs, ascending)

    avg_pairs = [
        (_display_score(model_avg[m], metric), ri)
        for ri, m in enumerate(sorted_models) if model_avg[m] is not None
    ]
    avg_best = _best_row(avg_pairs, ascending)

    # --- Build HTML ---
    tbl_id = "sum-" + _slug(task) + "-" + _slug(metric) + tbl_id_suffix

    def _td(val_str, is_best=False, is_missing=False, extra_attrs=""):
        if is_missing:
            style = f' style="background:{MISSING_COLOR}"'
        elif is_best:
            style = f' style="background:{HIGHLIGHT_COLOR}"'
        else:
            style = ""
        return f"<td{extra_attrs}{style}>{val_str}</td>"

    lines = []

    title = _format_suptitle("Summary \u00b7 " + task, metric)
    lines.append(
        f'<div style="font-size:15px;font-weight:600;color:#475569;margin:8px 0">{title}</div>'
    )

    lines.append(f'<table class="ov-tbl" id="{tbl_id}">')

    # Header
    lines.append("<thead><tr>")
    lines.append("<th>Model</th>")
    for lang in languages:
        grp = _slug(lang)
        if lang in expandable_langs:
            lines.append(
                f'<th>{lang} '
                f'<button class="toggle-btn" onclick="toggleCols(this,\'{tbl_id}\',\'{grp}\')">+</button></th>'
            )
            for ds in lang_datasets[lang]:
                lines.append(f'<th class="lang-col" data-group="{grp}">{ds}</th>')
        else:
            lines.append(f"<th>{lang}</th>")
    lines.append("<th>Average</th>")
    lines.append("</tr></thead>")

    # Body
    lines.append("<tbody>")
    for ri, m in enumerate(sorted_models):
        lines.append("<tr>")
        lines.append(f"<td>{m}</td>")

        for lang in languages:
            grp = _slug(lang)

            # Aggregate cell
            if m in lang_model_score[lang]:
                val = f"{_display_score(lang_model_score[lang][m], metric):.2f}"
                lines.append(_td(val, is_best=(ri == lang_best.get(lang))))
            else:
                lines.append(_td("-", is_missing=True))

            # Per-dataset sub-cells
            if lang in expandable_langs:
                for ds in lang_datasets[lang]:
                    attr = f' class="lang-col" data-group="{grp}"'
                    if m in lang_ds_model[lang][ds]:
                        val = f"{_display_score(lang_ds_model[lang][ds][m], metric):.2f}"
                        lines.append(_td(val, is_best=(ri == lang_ds_best[lang].get(ds)),
                                         extra_attrs=attr))
                    else:
                        lines.append(_td("-", is_missing=True, extra_attrs=attr))

        # Average
        if model_avg[m] is not None:
            val = f"{_display_score(model_avg[m], metric):.2f}"
            lines.append(_td(val, is_best=(ri == avg_best)))
        else:
            lines.append(_td("-", is_missing=True))

        lines.append("</tr>")

    lines.append("</tbody></table>")

    # JS — generic toggle (harmless if redefined by other tables)
    lines.append("""\
<script>
function toggleCols(btn, tblId, group) {
  var tbl = document.getElementById(tblId);
  var cells = tbl.querySelectorAll('[data-group="' + group + '"]');
  if (!cells.length) return;
  var show = cells[0].style.display !== 'table-cell';
  for (var i = 0; i < cells.length; i++)
    cells[i].style.display = show ? 'table-cell' : 'none';
  btn.textContent = show ? '\\u2212' : '+';
}
</script>""")

    collector.append({
        "category": "Summary \u00b7 " + task,
        "chart_type": "table",
        "metric": metric,
        "raw_html": "\n".join(lines),
    })


def _build_dual_summary_tables(task, metric, task_raw, agg_lang, agg_sub,
                                collector):
    """Build two summary tables (language & sub-task) with a toggle bar."""
    # Collect HTML from each view into a temporary list
    lang_collector = []
    _build_summary_table(task, metric, task_raw, agg_lang, lang_collector,
                         group_key_fn=None, tbl_id_suffix="-lang")
    sub_collector = []
    _build_summary_table(task, metric, task_raw, agg_sub, sub_collector,
                         group_key_fn=_effective_subtask, tbl_id_suffix="-sub")

    if not lang_collector and not sub_collector:
        return

    toggle_id = _slug(task) + "-" + _slug(metric)
    lines = []

    # Toggle bar CSS (only emitted once; harmless if repeated)
    lines.append("""\
<style>
.toggle-bar { display: inline-flex; gap: 0; margin: 8px 0; border-radius: 4px; overflow: hidden;
  border: 1px solid #3a5a8c; }
.toggle-bar button { padding: 4px 14px; font-size: 12px; font-weight: 600; cursor: pointer;
  border: none; background: #e2e8f0; color: #475569; transition: background .15s, color .15s; }
.toggle-bar button.active { background: #3a5a8c; color: white; }
</style>""")

    lines.append(f'<div class="toggle-bar" id="tbar-{toggle_id}">')
    lines.append(
        f'<button class="active" onclick="toggleSumView(\'{toggle_id}\',\'lang\')">Language</button>'
    )
    lines.append(
        f'<button onclick="toggleSumView(\'{toggle_id}\',\'sub\')">Sub-task</button>'
    )
    lines.append("</div>")

    # Language view (visible by default)
    lang_html = lang_collector[0]["raw_html"] if lang_collector else ""
    lines.append(f'<div id="sv-lang-{toggle_id}">{lang_html}</div>')

    # Sub-task view (hidden by default)
    sub_html = sub_collector[0]["raw_html"] if sub_collector else ""
    lines.append(f'<div id="sv-sub-{toggle_id}" style="display:none">{sub_html}</div>')

    # JS toggle
    lines.append("""\
<script>
function toggleSumView(id, view) {
  var langDiv = document.getElementById('sv-lang-' + id);
  var subDiv = document.getElementById('sv-sub-' + id);
  var bar = document.getElementById('tbar-' + id);
  if (!langDiv || !subDiv || !bar) return;
  var btns = bar.querySelectorAll('button');
  if (view === 'lang') {
    langDiv.style.display = '';
    subDiv.style.display = 'none';
    btns[0].classList.add('active');
    btns[1].classList.remove('active');
  } else {
    langDiv.style.display = 'none';
    subDiv.style.display = '';
    btns[0].classList.remove('active');
    btns[1].classList.add('active');
  }
}
</script>""")

    collector.append({
        "category": "Summary \u00b7 " + task,
        "chart_type": "table",
        "metric": metric,
        "raw_html": "\n".join(lines),
    })


# ---------------------------------------------------------------------------
# HTML Report Builder
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AudioBench Results</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         display: flex; min-height: 100vh; background: #f5f6f8; color: #222; }

  /* Sidebar */
  nav.sidebar { position: fixed; top: 0; left: 0; width: 240px; height: 100vh;
                overflow-y: auto; background: #1e293b; color: #cbd5e1; padding: 20px 0;
                z-index: 100; }
  nav.sidebar h2 { font-size: 15px; font-weight: 700; padding: 0 16px 14px; color: #f1f5f9;
                    border-bottom: 1px solid #334155; margin-bottom: 8px; }
  nav.sidebar ul { list-style: none; }
  nav.sidebar li a { display: block; padding: 7px 16px; font-size: 13px; color: #94a3b8;
                     text-decoration: none; transition: background .15s, color .15s; }
  nav.sidebar li a:hover, nav.sidebar li a.active { background: #334155; color: #e2e8f0; }
  nav.sidebar li.nav-group { font-size: 11px; font-weight: 700; text-transform: uppercase;
                              color: #64748b; padding: 14px 16px 4px; letter-spacing: .05em; }

  /* Main content */
  main { margin-left: 240px; padding: 28px 32px; flex: 1; max-width: calc(100vw - 240px); }

  /* Sections */
  section.category { margin-bottom: 36px; }
  section.category > h2 { font-size: 20px; color: #1e293b; border-bottom: 2px solid #3b82f6;
                           padding-bottom: 6px; margin-bottom: 16px; }
  details { margin-bottom: 20px; }
  details > summary { cursor: pointer; font-size: 15px; font-weight: 600; color: #475569;
                       padding: 6px 0; user-select: none; }
  details > summary:hover { color: #1e40af; }
  .figure-wrapper { margin: 12px 0; overflow-x: auto; }
</style>
</head>
<body>
<nav class="sidebar">
  <h2>AudioBench Results</h2>
  <ul>
__NAV_ITEMS__
  </ul>
</nav>
<main>
__SECTIONS__
</main>
</body>
</html>
"""


def _slug(text):
    """Turn a category name into a URL-safe anchor id."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', text).strip('_')


def build_html_report(collected_figures, output_path):
    """Assemble a single HTML report from collected Plotly figures.

    Figures are grouped by category, with violin plots shown before tables
    within each group.  The sidebar is split into **Overview** and **Summary**
    (flat task list) groups.
    """
    from collections import OrderedDict

    # Group figures by category, preserving insertion order
    categories = OrderedDict()
    for item in collected_figures:
        cat = item["category"]
        categories.setdefault(cat, []).append(item)

    # --- Classify categories into overview, summary, per-dataset ---
    _PREFIX = "Summary \u00b7 "
    overview_cats = []          # category_name
    summary_cats = []           # (task_label, category_name)

    for cat in categories:
        if cat == "Overview":
            overview_cats.append(cat)
        elif cat.startswith(_PREFIX):
            task = cat[len(_PREFIX):]
            summary_cats.append((task, cat))

    # --- Build nav HTML ---
    nav_lines = []

    if overview_cats:
        nav_lines.append('    <li class="nav-group">Overview</li>')
        for cat in overview_cats:
            slug = _slug(cat)
            nav_lines.append(f'    <li><a href="#cat-{slug}">All Tasks</a></li>')

    if summary_cats:
        nav_lines.append('    <li class="nav-group">Summary</li>')
        for task, cat in summary_cats:
            slug = _slug(cat)
            nav_lines.append(f'    <li><a href="#cat-{slug}">{task}</a></li>')

    # --- Build section HTML ---
    section_blocks = []
    fig_counter = 0

    for cat, items in categories.items():
        slug = _slug(cat)

        violins = [it for it in items if it["chart_type"] == "violin"]
        tables = [it for it in items if it["chart_type"] == "table"]

        section_html = f'<section class="category" id="cat-{slug}">\n  <h2>{cat}</h2>\n'

        for chart_label, chart_items in [("Score Distributions", violins), ("Tables", tables)]:
            if not chart_items:
                continue
            section_html += f'  <details open>\n    <summary>{chart_label}</summary>\n'
            for it in chart_items:
                if "raw_html" in it:
                    section_html += f'    <div class="figure-wrapper">{it["raw_html"]}</div>\n'
                else:
                    fig_counter += 1
                    div_id = f"fig-{fig_counter}"
                    fig_html = it["fig"].to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        div_id=div_id,
                    )
                    section_html += f'    <div class="figure-wrapper">{fig_html}</div>\n'
            section_html += '  </details>\n'

        section_html += '</section>'
        section_blocks.append(section_html)

    html = _HTML_TEMPLATE.replace('__NAV_ITEMS__', '\n'.join(nav_lines))
    html = html.replace('__SECTIONS__', '\n'.join(section_blocks))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    Path(output_path).write_text(html, encoding='utf-8')
    print(f"Saved report: {output_path} ({fig_counter} figures, {len(categories)} categories)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot AudioBench evaluation results as a single interactive HTML report."
    )
    parser.add_argument("input_folder", help="Path to results folder (e.g. results/)")
    parser.add_argument("--output_folder", type=str, default="plots/", help="Where to save report")
    args = parser.parse_args()

    # Load all scores
    entries = load_all_scores(args.input_folder)
    if not entries:
        print(f"No score files found in {args.input_folder}")
        return

    print(f"Loaded {len(entries)} score entries from {len({e['model_name'] for e in entries})} models")

    collector = []

    # --- Step 0: Overview table (all tasks × models) ---
    plot_overview_table(entries, collector)

    # --- Step 1: Summary violin plots (score distributions per task) ---
    for task, prefixes in sorted(AGGREGATE_DATASETS.items()):
        task_raw = [
            e for e in entries
            if any(e["dataset_name"].startswith(p) for p in prefixes)
        ]
        if task_raw:
            plot_violin_charts(task_raw, f"Summary \u00b7 {task}", collector)

    # --- Step 2: Summary tables (per-task, languages as columns, expandable per-dataset) ---
    plot_summary_tables(entries, collector)

    if not collector:
        print("No figures generated.")
        return

    output_path = os.path.join(args.output_folder, "report.html")
    build_html_report(collector, output_path)


if __name__ == "__main__":
    main()
