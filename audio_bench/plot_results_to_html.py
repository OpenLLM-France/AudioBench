#!/usr/bin/env python3
"""Plot AudioBench evaluation results as a single interactive HTML report.

Generates one HTML file with an Overview table (all tasks × models ranked)
and per-task Summary sections (language-column tables with expandable
per-dataset sub-columns, and optionally violin plots).

Output: {output_folder}/report.html

Usage examples:
    python src/plot_results_to_html.py results/
    python src/plot_results_to_html.py results/ --violin
    python src/plot_results_to_html.py results/ --output_folder my_plots/
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express.colors as pxcolors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOWER_IS_BETTER = {"wer"}
ZERO_TO_ONE_RANGE = {"wer", "meteor", "acc"}
_IGNORED_DATASETS = {"StressTest_SSR", "VoxCeleb-accent", "MuChoMusic"}

_MODEL_NAME_CORRECTIONS = {
    "LINAGORA/Canary-Qwen3-5B-Thinking": "LINAGORA/Canary-Qwen3-4B-v0-quick",
    "LINAGORA/Canary-Qwen3-1.7B-v2": "LINAGORA/Canary-Qwen3-1.7B-v0-quick",
}

_TASK_METRIC_OVERRIDE = {
    "AST": "meteor",
}

# Task display name overrides (raw task field → display name).
# Tasks not listed here use str.title() automatically.
_TASK_DISPLAY = {
    "ASR": "ASR",
    "AST": "AST",
}


# Patterns are tried in order; first regex match wins. Exact strings work too
# since they are compiled as regexes.
_MODEL_SIZE_OVERRIDES = [
    (r"^LINAGORA/Canary[-_]Qwen3[-_]1\.7B", "2.5B"),
    (r"^LINAGORA/Canary[-_]Qwen3[-_]4B", "4.8B"),
    (r"^LINAGORA/Canary_Luciole-1B", "2.1B"),
    (r"^microsoft/Phi-4-multimodal-instruct$", "5.6B"),
    (r"^nvidia/audio-flamingo-3-hf$", "8.2B"),
    (r"^Qwen/Qwen2-Audio-7B-Instruct$", "8.4B"),
    (r"^Qwen/Qwen2\.5-Omni-7B$", "11B"),
    (r"^Qwen/Qwen2\.5-Omni-3B$", "5.9B"),
    (r"^mistralai/Voxtral-Mini-3B-2507$", "4.68B"),
]
_MODEL_SIZE_OVERRIDES = [(re.compile(p), s) for p, s in _MODEL_SIZE_OVERRIDES]


def _task_display_name(raw_task: str) -> str:
    return _TASK_DISPLAY.get(raw_task.upper(), raw_task.title())


def _group_by_task(entries):
    """Group entries by their task field, returning {display_name: [entries]}."""
    groups = defaultdict(list)
    for e in entries:
        raw = e.get("task", "")
        if raw:
            groups[_task_display_name(raw)].append(e)
    return dict(sorted(groups.items()))


# Super-category mapping: raw task (upper-cased) → super-category label
_SUPER_CATEGORY = {
    "ASR": "ASR",
    "AST": "AST",
    "QUESTION ANSWERING": "QA",
    "MATH QUESTION ANSWERING": "QA",
    "MUSIC QUESTION ANSWERING": "Music",
    "MUSIC CAPTIONING": "Music",
    "AUDIO QUESTION ANSWERING": "Sound",
    "AUDIO CAPTIONING": "Sound",
}

_SUPER_CATEGORY_ORDER = ["ASR", "AST", "QA", "Others", "Music", "Sound"]


def _super_category(raw_task: str) -> str:
    return _SUPER_CATEGORY.get(raw_task.upper(), "Others")


def _group_by_super_category(entries):
    """Return OrderedDict {super_cat: {task_display: [entries]}}."""
    from collections import OrderedDict
    tmp = defaultdict(lambda: defaultdict(list))
    for e in entries:
        raw = e.get("task", "")
        if raw:
            sc = _super_category(raw)
            task = _task_display_name(raw)
            tmp[sc][task].append(e)
    result = OrderedDict()
    for sc in _SUPER_CATEGORY_ORDER:
        if sc in tmp:
            result[sc] = dict(sorted(tmp[sc].items()))
    for sc in sorted(tmp.keys()):
        if sc not in result:
            result[sc] = dict(sorted(tmp[sc].items()))
    return result

# Language display order: listed languages come first in this order,
# unlisted languages follow alphabetically, trailing languages come last.
_LANG_ORDER_HEAD = ["FR", "EN"]
_LANG_ORDER_TAIL = ["PT", "NL"]


def _lang_sort_key(lang: str):
    """Return a tuple that sorts languages per the configured order."""
    up = (lang or "").upper()
    if up in _LANG_ORDER_HEAD:
        return (0, _LANG_ORDER_HEAD.index(up), up)
    if up in _LANG_ORDER_TAIL:
        return (2, _LANG_ORDER_TAIL.index(up), up)
    # For language pairs (e.g. "FR-EN"), sort by source language prefix
    prefix = up.split("-")[0] if "-" in up else None
    if prefix and prefix in _LANG_ORDER_HEAD:
        return (0, _LANG_ORDER_HEAD.index(prefix), up)
    return (1, 0, up)


HIGHLIGHT_COLOR = "#b0c1d7"
MISSING_COLOR = "#e0e0e0"
RANK_COLORS = {
    "first": "#5dade2",        # sky blue
    "second": "#82e0aa",       # green
    "last": "#e74c3c",         # bold red
    "before_last": "#fdcb6e",  # amber
}

# Language grouping for the Languages navigation section
LANGUAGE_GROUPS = {
    "French":  {"FR", "FR-EN", "FR-ES"},
    "English": {"EN"},
    "Others":  None,  # catch-all
}

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
        model_id = model_dir.name

        for filepath in sorted(model_dir.rglob("*_score.json")):
            dataset_name = filepath.name.removesuffix("_score.json")
            if dataset_name in _IGNORED_DATASETS:
                continue

            try:
                data = json.loads(filepath.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            metrics = data.get("metrics", [])
            if not metrics:
                continue

            model_name = data.get("model_name", model_id)
            model_name = _MODEL_NAME_CORRECTIONS.get(model_name, model_name)
            task = data.get("task")
            language = data.get("language")
            sub_task = data.get("sub_task")

            for metric_name in metrics:
                try:
                    raw_score = data[metric_name]
                except (KeyError, TypeError):
                    continue

                # Metrics may store scores as dicts: new format has "score" key,
                # old judge format has "judge_score" key
                if isinstance(raw_score, dict):
                    score = raw_score.get("score", raw_score.get("judge_score"))
                    if score is None:
                        continue
                    all_scores = raw_score.get("all_scores")  # list[float] or None
                    std = raw_score.get("std")                 # float or None
                    n = len(all_scores) if all_scores else None
                else:
                    score = raw_score  # old bare-float format
                    all_scores = std = n = None

                if not isinstance(score, (int, float)):
                    continue

                entry = {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "metric_name": metric_name,
                    "score": float(score),
                    "task": task,
                    "language": language,
                    "sub_task": sub_task,
                }
                if all_scores is not None:
                    entry["all_scores"] = all_scores
                if std is not None:
                    entry["std"] = float(std)
                if n is not None:
                    entry["n"] = int(n)
                entries.append(entry)

    return entries

# ---------------------------------------------------------------------------
# Display-name helpers
# ---------------------------------------------------------------------------

def _dataset_display_name(entry):
    """Return a display name for per-dataset breakdowns.

    For AST entries, includes the language pair (e.g. 'Multilingual_TEDx (FR→EN)').
    """
    name = entry["dataset_name"]
    if entry.get("task") == "AST" and entry.get("language"):
        lang = entry["language"]
        parts = lang.split("-")
        if len(parts) == 2:
            name = f"{name} ({parts[0]}→{parts[1]})"
        else:
            name = f"{name} ({lang})"
    return name

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

    task_groups = _group_by_task(entries)
    if task_filter:
        tf = task_filter.upper()
        task_groups = {k: v for k, v in task_groups.items() if k.upper() == tf}

    all_models = sorted({e["model_name"] for e in entries})
    all_metrics = sorted({e["metric_name"] for e in entries})

    for agg_task, task_entries in task_groups.items():

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

        for lang_key, lang_entries in sorted(lang_groups.items(), key=lambda x: _lang_sort_key(x[0])):
            for metric in all_metrics:
                for model in all_models:
                    matching = [
                        e for e in lang_entries
                        if e["model_name"] == model and e["metric_name"] == metric
                    ]
                    scores = [e["score"] for e in matching]

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
                            for e in matching
                        }
                        lang = languages.pop() if len(languages) == 1 else "MIXED"
                        label = agg_task

                    agg_entry = {
                        "model_name": model,
                        "dataset_name": label,
                        "metric_name": metric,
                        "score": sum(scores) / len(scores),
                        "task": agg_task,
                        "language": lang,
                    }

                    # Pool per-sample scores from child entries for CI
                    pooled = []
                    for e in matching:
                        if "all_scores" in e:
                            pooled.extend(e["all_scores"])
                    if pooled:
                        agg_entry["all_scores"] = pooled
                        agg_entry["std"] = float(np.std(np.array(pooled)))
                        agg_entry["n"] = len(pooled)

                    aggregated.append(agg_entry)

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


def _compute_ci(std, n):
    """Compute 95% confidence interval half-width, or None."""
    if std is None or n is None or n <= 0:
        return None
    return 1.96 * std / math.sqrt(n)


def _format_score_with_ci(score, metric, std=None, n=None):
    """Return (html_str, tooltip_str) with optional CI display.

    html_str:   '18.50 <span class="ci">±0.32</span>'  (or just '18.50')
    tooltip_str: '18.50 [18.18, 18.82], n=676'          (or just '18.50')
    """
    disp = _display_score(score, metric)
    base = f"{disp:.2f}"
    ci = _compute_ci(std, n)
    if ci is not None:
        # Scale CI the same way as the score display
        ci_disp = ci * 100 if metric in ZERO_TO_ONE_RANGE else ci
        lo = disp - ci_disp
        hi = disp + ci_disp
        html_str = f'{base} <span class="ci">\u00b1{ci_disp:.2f}</span>'
        tooltip_str = f"{base} [{lo:.2f}, {hi:.2f}], n={n}"
    else:
        html_str = base
        tooltip_str = base
    return html_str, tooltip_str


def _classify_language(lang_str):
    """Classify a language string into a LANGUAGE_GROUPS key."""
    lang = (lang_str or "UNKNOWN").upper()
    for group_name, lang_set in LANGUAGE_GROUPS.items():
        if lang_set is not None and lang in lang_set:
            return group_name
    return "Others"


def _sort_ascending(metric):
    """Return True if lower is better for this metric."""
    return metric in LOWER_IS_BETTER


def _td(val_str, is_best=False, is_missing=False, extra_attrs="", title="", rank_key=None):
    """Build a <td> element with optional highlight/missing styling."""
    if is_missing:
        style = f' style="background:{MISSING_COLOR}"'
    elif rank_key and rank_key in RANK_COLORS:
        style = f' style="background:{RANK_COLORS[rank_key]}"'
    elif is_best:
        style = f' style="background:{HIGHLIGHT_COLOR}"'
    else:
        style = ""
    title_attr = f' title="{title}"' if title else ""
    return f"<td{extra_attrs}{style}{title_attr}>{val_str}</td>"

def _extract_model_size(model_name):
    """Extract size like '7B' from model name by matching patterns like '_7b' or '_3b_'."""
    for pattern, size in _MODEL_SIZE_OVERRIDES:
        if pattern.search(model_name):
            return size
    match = re.search(r'(\d+)[bB](?:_|$)', model_name)
    return f"{match.group(1)}B" if match else ""


def _best_row(pairs, asc):
    """Return the row index of the best value, or None if *pairs* is empty."""
    if not pairs:
        return None
    return (min if asc else max)(pairs, key=lambda x: x[0])[1]


def _ranked_rows(pairs, asc):
    """Return {row_index: rank_key} for 1st, 2nd, last, before-last positions.

    *pairs* is a list of (value, row_index). *asc* True means lower is better.
    """
    if not pairs:
        return {}
    ranked = sorted(pairs, key=lambda x: x[0], reverse=not asc)
    result = {}
    result[ranked[0][1]] = "first"
    if len(ranked) >= 2:
        result[ranked[1][1]] = "second"
    if len(ranked) >= 3:
        result[ranked[-1][1]] = "last"
    if len(ranked) >= 4:
        result[ranked[-2][1]] = "before_last"
    return result


def _most_common_metric(entries):
    """Return the most frequent metric_name among *entries*."""
    counts = defaultdict(int)
    for e in entries:
        counts[e["metric_name"]] += 1
    return max(counts, key=counts.get)


def _compute_normalized_scores(all_models, item_model_score, item_ascending):
    """Compute min-max and z-score normalized aggregate scores.

    Parameters
    ----------
    all_models : list of str
    item_model_score : dict
        item -> model -> score (float).  Items can be super-categories, tasks,
        languages, etc.
    item_ascending : dict
        item -> bool.  True means lower is better (e.g. WER).

    Returns (model_minmax, model_zscore) dicts.
    """
    # Convert to higher-is-better per item and pre-compute per-item stats
    item_model_hib = {}
    item_stats = {}  # item -> (lo, hi, mean, std)
    for item, scores in item_model_score.items():
        asc = item_ascending[item]
        hib = {m: (100.0 - s) if asc else s for m, s in scores.items()}
        item_model_hib[item] = hib
        vals = np.array(list(hib.values()))
        item_stats[item] = (float(vals.min()), float(vals.max()),
                            float(vals.mean()), float(vals.std()))

    model_minmax = {}
    model_zscore = {}
    for m in all_models:
        norm_scores = []
        z_scores = []
        for item, hib in item_model_hib.items():
            if m not in hib:
                continue
            lo, hi, mean, std = item_stats[item]
            v = hib[m]
            norm_scores.append((v - lo) / (hi - lo) if hi > lo else 1.0)
            z_scores.append((v - mean) / std if std > 0 else 0.0)
        model_minmax[m] = sum(norm_scores) / len(norm_scores) if norm_scores else float("-inf")
        model_zscore[m] = sum(z_scores) / len(z_scores) if z_scores else float("-inf")

    return model_minmax, model_zscore


def _agg_columns_html(table_aggregates, sorted_models, aggregate_values):
    """Build ranked-row dicts and per-row rendering info for aggregate columns.

    Parameters
    ----------
    table_aggregates : list of str
        Which aggregates to include (keys into ``_AGG_META``).
    sorted_models : list of str
    aggregate_values : dict
        Mapping aggregate name -> model -> value.

    Returns *agg_render* dict keyed by aggregate name.
    """
    agg_render = {}
    for name in table_aggregates:
        meta = _AGG_META[name]
        values = aggregate_values[name]
        agg_render[name] = {
            "ranks": _ranked_rows(
                [(values[m], ri) for ri, m in enumerate(sorted_models)
                 if values[m] != meta["sentinel"]],
                asc=not meta["higher_is_better"],
            ),
            "values": values,
            "sentinel": meta["sentinel"],
            "fmt": meta["fmt"],
        }
    return agg_render


def _sort_models_by_aggregate(models, aggregate_values, agg_name):
    """Sort *models* by the aggregate named *agg_name* (best first).

    Uses ``_AGG_META`` to determine sort direction and sentinel value.
    """
    meta = _AGG_META[agg_name]
    values = aggregate_values[agg_name]
    sentinel = meta["sentinel"]
    higher = meta["higher_is_better"]
    return sorted(
        models,
        key=lambda m: (values[m] == sentinel, values[m]),
        reverse=higher,
    )


def _sort_models_by_avg(models, score_fn, ascending):
    """Sort models by average score. *score_fn(model)* -> list of scores."""
    model_avg = {}
    for m in models:
        scores = score_fn(m)
        model_avg[m] = sum(scores) / len(scores) if scores else None
    sorted_models = sorted(
        models,
        key=lambda m: (model_avg[m] is None,
                       model_avg[m] if model_avg[m] is not None else 0),
        reverse=not ascending,
    )
    return sorted_models, model_avg


_TOGGLE_COLS_JS = """\
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
</script>"""


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
    """Yield (metric, ds_model_score, datasets, models, ds_model_entry) per metric.

    *ds_model_entry* maps dataset -> model -> full entry dict (for all_scores access).
    """
    by_metric = group_entries_by_metric(entries)
    for metric, metric_entries in sorted(by_metric.items()):
        ds_model_score = defaultdict(dict)
        ds_model_entry = defaultdict(dict)
        for e in metric_entries:
            ds_model_score[e["dataset_name"]][e["model_name"]] = e["score"]
            ds_model_entry[e["dataset_name"]][e["model_name"]] = e
        datasets = sorted(ds_model_score.keys())
        models = sorted({e["model_name"] for e in metric_entries})
        if datasets and models:
            yield metric, ds_model_score, datasets, models, ds_model_entry


def _format_suptitle(title_prefix, metric):
    """Produce a consistent, unit-aware suptitle."""
    unit = " (%)" if metric in ZERO_TO_ONE_RANGE else ""
    return f"{title_prefix} — {metric.upper()}{unit}"


# ---------------------------------------------------------------------------
# Plotting — Violin Charts
# ---------------------------------------------------------------------------

def plot_violin_charts(entries, title_prefix, collector):
    """Produce violin plots showing per-dataset score distributions per model.

    When entries carry ``all_scores`` (per-sample data), those individual scores
    are pooled across datasets for a rich distribution.  Otherwise, falls back to
    one point per dataset aggregate with jittered scatter.
    """
    all_models = sorted({e["model_name"] for e in entries})
    color_map = _model_color_map(all_models)

    for metric, ds_model_score, datasets, models, ds_model_entry in _prepare_metric_data(entries):
        ascending = _sort_ascending(metric)

        # Sort models by average score (best first)
        sorted_models, model_avg = _sort_models_by_avg(
            models,
            lambda m: [ds_model_score[ds][m] for ds in datasets if m in ds_model_score[ds]],
            ascending,
        )

        fig = go.Figure()

        for m in sorted_models:
            # Collect per-sample scores (rich) OR per-dataset aggregates (sparse)
            rich_values = []
            sparse_values = []
            hover_texts = []
            for ds in datasets:
                if m not in ds_model_score[ds]:
                    continue
                e = ds_model_entry[ds][m]
                if "all_scores" in e and e["all_scores"]:
                    rich_values.extend(e["all_scores"])
                else:
                    sparse_values.append(ds_model_score[ds][m])
                    hover_texts.append(
                        f"{ds}: {_display_score(ds_model_score[ds][m], metric):.2f}"
                    )

            has_rich = len(rich_values) > 0
            if has_rich:
                display_vals = [_display_score(s, metric) for s in rich_values]
            else:
                display_vals = [_display_score(s, metric) for s in sparse_values]

            # Cap display values to prevent outliers from distorting the plot
            display_vals = [min(v, 100) for v in display_vals]

            if has_rich and len(display_vals) > 50:
                fig.add_trace(go.Violin(
                    y=display_vals,
                    name=m,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    marker=dict(color=color_map[m], size=5, opacity=0.7),
                    line=dict(color=color_map[m]),
                    fillcolor=color_map[m],
                    opacity=0.5,
                    hoverinfo="y+name",
                    showlegend=False,
                ))
            else:
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
                    hovertext=hover_texts if not has_rich else None,
                    hoverinfo="text+name" if not has_rich else "y+name",
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
            yaxis_range=[0, 100] if metric in ZERO_TO_ONE_RANGE else None,
        )

        collector.append({
            "category": title_prefix,
            "chart_type": "violin",
            "metric": metric,
            "fig": fig,
        })

# ---------------------------------------------------------------------------
# Plotting — Size vs Performance scatter
# ---------------------------------------------------------------------------

def _compute_overview_ranks(entries, allowed_super_cats=None, sort_by="avg_rank"):
    """Compute per-model average rank across super-categories.

    This is the shared rank computation used by both the overview table and the
    size-vs-performance scatter plot.  It handles ``expandable_dataset``
    super-categories (e.g. QA) by averaging per-dataset scores rather than
    per-task scores, so that the ranking is consistent everywhere.

    Returns a dict with keys:

    * ``agg`` – aggregated entries (by_language=False)
    * ``task_entries`` – task -> [entries]
    * ``task_metric`` – task -> chosen metric name
    * ``tasks`` – sorted task list
    * ``all_models`` – sorted model list
    * ``task_model_score`` – task -> model -> (score, std, n)
    * ``sc_tasks`` – super_cat -> [task names]
    * ``super_cats`` – ordered list of super-categories
    * ``expandable_sc`` – set of SCs expandable by sub-task
    * ``expandable_dataset`` – set of SCs expandable by dataset
    * ``expandable_lang`` – set of SCs expandable by language
    * ``task_languages`` – task -> sorted list of languages
    * ``task_lang_scores`` – task -> lang -> model -> (score, std, n)
    * ``task_lang_datasets`` – task -> lang -> [dataset display names]
    * ``sc_datasets`` – sc -> [dataset display names]
    * ``sc_dataset_scores`` – sc -> dataset -> model -> (score, std, n)
    * ``sc_dataset_metric`` – sc -> dataset -> metric
    * ``sc_model_score`` – sc -> model -> display_score (float)
    * ``sc_ascending`` – sc -> bool
    * ``sc_model_rank`` – sc -> model -> rank (1-based)
    * ``task_model_rank`` – task -> model -> rank (1-based)
    * ``model_avg_rank`` – model -> avg rank (float, or ``float('inf')``)
    * ``model_minmax`` – model -> min-max normalized score (float, or ``float('-inf')``)
    * ``model_zscore`` – model -> z-score normalized score (float, or ``float('-inf')``)
    * ``sorted_models`` – models sorted by avg rank
    """
    agg = aggregate_entries(entries, by_language=False)
    if not agg:
        return None

    # Group by task
    task_entries = defaultdict(list)
    for e in agg:
        task_entries[e["task"]].append(e)

    # For each task pick the most common metric
    task_metric = {}
    for task, ents in task_entries.items():
        override = _TASK_METRIC_OVERRIDE.get(task.upper())
        if override and any(e["metric_name"] == override for e in ents):
            task_metric[task] = override
        else:
            task_metric[task] = _most_common_metric(ents)

    tasks = sorted(task_metric.keys())
    if not tasks:
        return None

    all_models = sorted({e["model_name"] for e in agg})

    # Build score lookup: task -> model -> (score, std, n)
    task_model_score = {}
    for task in tasks:
        metric = task_metric[task]
        model_scores = {}
        for m in all_models:
            matching = [
                e for e in task_entries[task]
                if e["model_name"] == m and e["metric_name"] == metric
            ]
            scores = [e["score"] for e in matching]
            if scores:
                pooled = []
                for e in matching:
                    if "all_scores" in e:
                        pooled.extend(e["all_scores"])
                avg = sum(scores) / len(scores)
                if pooled:
                    model_scores[m] = (avg, float(np.std(np.array(pooled))), len(pooled))
                else:
                    model_scores[m] = (avg, None, None)
        task_model_score[task] = model_scores

    # --- Super-category grouping ---
    sc_tasks = defaultdict(list)
    for task in tasks:
        sc = _super_category(task)
        sc_tasks[sc].append(task)

    # Ordered super-categories
    super_cats = []
    for sc in _SUPER_CATEGORY_ORDER:
        if sc in sc_tasks:
            super_cats.append(sc)
    for sc in sorted(sc_tasks.keys()):
        if sc not in super_cats:
            super_cats.append(sc)

    if allowed_super_cats is not None:
        super_cats = [sc for sc in super_cats if sc in allowed_super_cats]

    expandable_sc = {sc for sc in super_cats if len(sc_tasks[sc]) > 1}
    expandable_sc.discard("QA")
    expandable_dataset = {sc for sc in ("QA",) if sc in sc_tasks}

    # --- Per-task language breakdown ---
    agg_lang = aggregate_entries(entries, by_language=True)
    task_lang_scores = defaultdict(lambda: defaultdict(dict))
    task_languages = defaultdict(set)
    for e in agg_lang:
        task = e["task"]
        if task not in task_metric:
            continue
        metric = task_metric[task]
        if e["metric_name"] != metric:
            continue
        lang = e["dataset_name"]
        m = e["model_name"]
        task_lang_scores[task][lang][m] = (e["score"], e.get("std"), e.get("n"))
        task_languages[task].add(lang)
    task_languages = {t: sorted(langs, key=_lang_sort_key) for t, langs in task_languages.items()}

    # Build task -> lang -> [dataset display names] for header labels
    task_lang_datasets = defaultdict(lambda: defaultdict(set))
    for e in entries:
        task = _task_display_name(e.get("task", ""))
        if task not in task_metric:
            continue
        lang = (e.get("language") or "UNKNOWN").upper()
        task_lang_datasets[task][lang].add(_dataset_display_name(e))
    task_lang_datasets = {
        t: {l: sorted(ds) for l, ds in langs.items()}
        for t, langs in task_lang_datasets.items()
    }

    # ASR/AST with >=2 languages get per-language expand
    expandable_lang = {
        sc for sc in super_cats
        if sc not in expandable_sc and sc not in expandable_dataset
        and sc in ("ASR", "AST")
        and len(task_languages.get(sc_tasks[sc][0], [])) >= 2
    }

    # Other multi-language SCs get per-dataset expand
    for sc in super_cats:
        if (sc not in expandable_sc and sc not in expandable_dataset
                and sc not in expandable_lang):
            task = sc_tasks[sc][0] if len(sc_tasks[sc]) == 1 else None
            if task and len(task_languages.get(task, [])) >= 2:
                expandable_dataset.add(sc)

    # --- Per-dataset breakdown for expandable_dataset SCs ---
    sc_datasets = {}
    sc_dataset_scores = {}
    sc_dataset_metric = {}

    for sc in expandable_dataset:
        datasets = []
        dataset_scores = defaultdict(dict)
        dataset_metric = {}
        multi_lang = any(
            len(task_languages.get(t, [])) >= 2 for t in sc_tasks[sc]
        )

        for task in sc_tasks[sc]:
            metric = task_metric[task]
            grouped = defaultdict(list)
            ds_lang = {}
            for e in entries:
                if (_task_display_name(e.get("task", "")) == task
                        and e["metric_name"] == metric):
                    grouped[(e["dataset_name"], e["model_name"])].append(e)
                    if e["dataset_name"] not in ds_lang:
                        ds_lang[e["dataset_name"]] = (e.get("language") or "UNKNOWN").upper()
            ds_names = sorted({k[0] for k in grouped})
            for ds in ds_names:
                display = f"{ds_lang.get(ds, '')} \u00b7 {ds}" if multi_lang else ds
                datasets.append(display)
                dataset_metric[display] = metric
                for m in all_models:
                    ds_entries = grouped.get((ds, m))
                    if ds_entries:
                        score = sum(e["score"] for e in ds_entries) / len(ds_entries)
                        pooled = []
                        for e in ds_entries:
                            if "all_scores" in e:
                                pooled.extend(e["all_scores"])
                        if pooled:
                            dataset_scores[display][m] = (score, float(np.std(np.array(pooled))), len(pooled))
                        else:
                            dataset_scores[display][m] = (score, None, None)

        sc_datasets[sc] = sorted(datasets) if multi_lang else datasets
        sc_dataset_scores[sc] = dataset_scores
        sc_dataset_metric[sc] = dataset_metric

    # Build super-category aggregate scores
    sc_model_score = {}
    for sc in super_cats:
        sc_model_score[sc] = {}
        if sc in expandable_dataset:
            for m in all_models:
                disp_scores = []
                for ds in sc_datasets[sc]:
                    if m in sc_dataset_scores[sc][ds]:
                        metric = sc_dataset_metric[sc][ds]
                        disp_scores.append(
                            _display_score(sc_dataset_scores[sc][ds][m][0], metric)
                        )
                if disp_scores:
                    sc_model_score[sc][m] = sum(disp_scores) / len(disp_scores)
        else:
            for m in all_models:
                task_disp_scores = []
                for task in sc_tasks[sc]:
                    if m in task_model_score[task]:
                        metric = task_metric[task]
                        task_disp_scores.append(
                            _display_score(task_model_score[task][m][0], metric)
                        )
                if task_disp_scores:
                    sc_model_score[sc][m] = sum(task_disp_scores) / len(task_disp_scores)

    # Per-super-category ranks
    sc_ascending = {}
    for sc in super_cats:
        sc_ascending[sc] = all(_sort_ascending(task_metric[t]) for t in sc_tasks[sc])

    sc_model_rank = {}
    for sc in super_cats:
        asc = sc_ascending[sc]
        scores = sc_model_score[sc]
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=not asc)
        sc_model_rank[sc] = {m: rank + 1 for rank, (m, _) in enumerate(ranked)}

    # Per-task ranks
    task_model_rank = {}
    for task in tasks:
        ascending = _sort_ascending(task_metric[task])
        scores = task_model_score[task]
        ranked = sorted(scores.items(), key=lambda x: x[1][0], reverse=not ascending)
        task_model_rank[task] = {m: rank + 1 for rank, (m, _) in enumerate(ranked)}

    # Average rank per model
    model_avg_rank = {}
    for m in all_models:
        ranks = [sc_model_rank[sc][m] for sc in super_cats if m in sc_model_rank[sc]]
        model_avg_rank[m] = sum(ranks) / len(ranks) if ranks else float("inf")

    # --- Normalized aggregate scores (Min-Max and Z-Score) ---
    model_minmax, model_zscore = _compute_normalized_scores(
        all_models, sc_model_score, sc_ascending,
    )

    agg_values = {"avg_rank": model_avg_rank, "minmax": model_minmax, "zscore": model_zscore}
    sorted_models = _sort_models_by_aggregate(all_models, agg_values, sort_by)

    return {
        "agg": agg,
        "task_entries": task_entries,
        "task_metric": task_metric,
        "tasks": tasks,
        "all_models": all_models,
        "task_model_score": task_model_score,
        "sc_tasks": sc_tasks,
        "super_cats": super_cats,
        "expandable_sc": expandable_sc,
        "expandable_dataset": expandable_dataset,
        "expandable_lang": expandable_lang,
        "task_languages": task_languages,
        "task_lang_scores": task_lang_scores,
        "task_lang_datasets": task_lang_datasets,
        "sc_datasets": sc_datasets,
        "sc_dataset_scores": sc_dataset_scores,
        "sc_dataset_metric": sc_dataset_metric,
        "sc_model_score": sc_model_score,
        "sc_ascending": sc_ascending,
        "sc_model_rank": sc_model_rank,
        "task_model_rank": task_model_rank,
        "model_avg_rank": model_avg_rank,
        "model_minmax": model_minmax,
        "model_zscore": model_zscore,
        "sorted_models": sorted_models,
    }


AGGREGATE_MEASURES = ["minmax", "zscore", "avg_rank"]

_AGG_META = {
    "avg_rank": {
        "label": "Avg Rank",
        "y_label": "Average Rank (lower is better)",
        "key": "model_avg_rank",
        "higher_is_better": False,
        "sentinel": float("inf"),
        "fmt": ".1f",
    },
    "minmax": {
        "label": "Min-Max",
        "y_label": "Min-Max Normalized Score (higher is better)",
        "key": "model_minmax",
        "higher_is_better": True,
        "sentinel": float("-inf"),
        "fmt": ".3f",
    },
    "zscore": {
        "label": "Z-Score",
        "y_label": "Z-Score Normalized Score (higher is better)",
        "key": "model_zscore",
        "higher_is_better": True,
        "sentinel": float("-inf"),
        "fmt": ".2f",
    },
}


def plot_size_vs_performance(entries, collector, *, category="Overview",
                             overview_data=None, figure_aggregates=None):
    """Scatter plot(s) of aggregate score vs model size.

    Generates one figure per measure in *figure_aggregates*.
    If *overview_data* is provided, reuses it instead of recomputing.
    """
    if figure_aggregates is None:
        figure_aggregates = ["avg_rank"]

    data = overview_data or _compute_overview_ranks(entries)
    if data is None:
        return

    all_models = data["all_models"]
    color_map = _model_color_map(all_models)

    for agg_name in figure_aggregates:
        meta = _AGG_META[agg_name]
        model_values = data[meta["key"]]
        sentinel = meta["sentinel"]

        xs, ys, labels = [], [], []
        for m in all_models:
            size_str = _extract_model_size(m)
            if not size_str or model_values[m] == sentinel:
                continue
            xs.append(float(size_str.rstrip("B")))
            ys.append(model_values[m])
            labels.append(m)

        if not xs:
            continue

        colors = [color_map.get(m, "#888") for m in labels]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=12, color=colors, line=dict(width=1, color="#333")),
            hovertemplate=f"%{{text}}<br>Size: %{{x}}B<br>{meta['label']}: %{{y:{meta['fmt']}}}<extra></extra>",
        ))

        fig.update_layout(
            title_text=f"Performance ({meta['label']}) vs Model Size",
            title_font_size=16,
            xaxis_title="Model Size (B parameters)",
            yaxis_title=meta["y_label"],
            yaxis_autorange="reversed" if not meta["higher_is_better"] else None,
            height=500,
            width=800,
            template="plotly_white",
        )

        collector.append({
            "category": category,
            "chart_type": "table",
            "metric": f"size_vs_perf_{agg_name}",
            "raw_html": fig.to_html(full_html=False, include_plotlyjs=False),
        })


# ---------------------------------------------------------------------------
# Plotting — Overview Table (all tasks × models)
# ---------------------------------------------------------------------------

def plot_overview_table(entries, collector, *, title="Overview",
                        table_id="overview-tbl", allowed_super_cats=None,
                        table_aggregates=None):
    """Build an overview HTML table with super-category columns.

    Each super-category column shows the average score across its tasks.
    Super-categories with multiple tasks have a [+] toggle to reveal
    individual task sub-columns.

    Parameters
    ----------
    allowed_super_cats : set or None
        When set, only super-categories in this set are included.
    table_aggregates : list of str or None
        Which aggregate columns to show (from AGGREGATE_MEASURES).
        Defaults to all.

    Returns the computed overview data dict so callers can reuse it
    (e.g. to pass to ``plot_size_vs_performance``).
    """
    if table_aggregates is None:
        table_aggregates = list(AGGREGATE_MEASURES)

    data = _compute_overview_ranks(entries, allowed_super_cats=allowed_super_cats,
                                    sort_by=table_aggregates[0])
    if data is None:
        return None

    agg = data["agg"]
    task_entries = data["task_entries"]
    task_metric = data["task_metric"]
    tasks = data["tasks"]
    all_models = data["all_models"]
    task_model_score = data["task_model_score"]
    sc_tasks = data["sc_tasks"]
    super_cats = data["super_cats"]
    expandable_sc = data["expandable_sc"]
    expandable_dataset = data["expandable_dataset"]
    expandable_lang = data["expandable_lang"]
    task_languages = data["task_languages"]
    task_lang_scores = data["task_lang_scores"]
    task_lang_datasets = data["task_lang_datasets"]
    sc_datasets = data["sc_datasets"]
    sc_dataset_scores = data["sc_dataset_scores"]
    sc_dataset_metric = data["sc_dataset_metric"]
    sc_model_score = data["sc_model_score"]
    sc_ascending = data["sc_ascending"]
    sc_model_rank = data["sc_model_rank"]
    task_model_rank = data["task_model_rank"]
    model_avg_rank = data["model_avg_rank"]
    model_minmax = data["model_minmax"]
    model_zscore = data["model_zscore"]
    sorted_models = data["sorted_models"]

    # --- Ranked row indices for highlighting ---
    sc_ranks = {}
    for sc in super_cats:
        asc = sc_ascending[sc]
        pairs = [
            (sc_model_score[sc][m], ri)
            for ri, m in enumerate(sorted_models) if m in sc_model_score[sc]
        ]
        sc_ranks[sc] = _ranked_rows(pairs, asc)

    task_ranks = {}
    for task in tasks:
        metric = task_metric[task]
        asc = _sort_ascending(metric)
        pairs = [
            (_display_score(task_model_score[task][m][0], metric), ri)
            for ri, m in enumerate(sorted_models) if m in task_model_score[task]
        ]
        task_ranks[task] = _ranked_rows(pairs, asc)

    # Ranked rows for per-language sub-columns (expandable_lang SCs)
    task_lang_ranks = defaultdict(dict)
    for sc in expandable_lang:
        task = sc_tasks[sc][0]
        metric = task_metric[task]
        asc = _sort_ascending(metric)
        for lang in task_languages.get(task, []):
            pairs = [
                (_display_score(task_lang_scores[task][lang][m][0], metric), ri)
                for ri, m in enumerate(sorted_models) if m in task_lang_scores[task][lang]
            ]
            task_lang_ranks[task][lang] = _ranked_rows(pairs, asc)

    # Ranked rows for per-dataset sub-columns (expandable_dataset SCs)
    sc_dataset_ranks = defaultdict(dict)
    for sc in expandable_dataset:
        for ds in sc_datasets[sc]:
            metric = sc_dataset_metric[sc][ds]
            ds_asc = _sort_ascending(metric)
            pairs = [
                (_display_score(sc_dataset_scores[sc][ds][m][0], metric), ri)
                for ri, m in enumerate(sorted_models) if m in sc_dataset_scores[sc][ds]
            ]
            sc_dataset_ranks[sc][ds] = _ranked_rows(pairs, ds_asc)

    agg_render = _agg_columns_html(table_aggregates, sorted_models, {
        "avg_rank": model_avg_rank, "minmax": model_minmax, "zscore": model_zscore,
    })

    # --- Build HTML ---
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
.ci {{ font-size: 0.75em; color: #64748b; }}
</style>""")

    # Color legend
    lines.append(
        '<div style="display:flex;gap:16px;align-items:center;font-size:12px;margin:8px 0;">'
        f'<span style="display:inline-flex;align-items:center;gap:4px;">'
        f'<span style="width:12px;height:12px;background:{RANK_COLORS["first"]};border:1px solid #ccc;border-radius:2px;"></span>1st</span>'
        f'<span style="display:inline-flex;align-items:center;gap:4px;">'
        f'<span style="width:12px;height:12px;background:{RANK_COLORS["second"]};border:1px solid #ccc;border-radius:2px;"></span>2nd</span>'
        f'<span style="display:inline-flex;align-items:center;gap:4px;">'
        f'<span style="width:12px;height:12px;background:{RANK_COLORS["before_last"]};border:1px solid #ccc;border-radius:2px;"></span>Second to last</span>'
        f'<span style="display:inline-flex;align-items:center;gap:4px;">'
        f'<span style="width:12px;height:12px;background:{RANK_COLORS["last"]};border:1px solid #ccc;border-radius:2px;"></span>Last</span>'
        '</div>'
    )

    lines.append(f'<table class="ov-tbl" id="{table_id}">')

    # --- Header ---
    lines.append("<thead><tr>")
    lines.append("<th>Model</th>")
    lines.append("<th>Size</th>")
    for agg_name in table_aggregates:
        lines.append(f"<th>{_AGG_META[agg_name]['label']}</th>")
    for sc in super_cats:
        sc_slug = _slug(sc)
        cat_label = "Tasks \u00b7 " + sc
        section_anchor = f"cat-{_slug(cat_label)}"

        if sc in expandable_sc:
            # Multi-task SC: show SC name with [+], hidden task sub-columns
            task_metrics_label = ", ".join(
                f"{task_metric[t].upper()}" for t in sc_tasks[sc]
            )
            label = f'<a href="#{section_anchor}">{sc}</a>'
            lines.append(
                f'<th>{label} '
                f'<button class="toggle-btn" onclick="toggleOvSc(this,\'{sc_slug}\')">+</button></th>'
            )
            for task in sc_tasks[sc]:
                metric = task_metric[task]
                unit = " %" if metric in ZERO_TO_ONE_RANGE else ""
                lines.append(
                    f'<th class="lang-col" data-sc="{sc_slug}">'
                    f'{task} ({metric.upper()}{unit})</th>'
                )
        elif sc in expandable_dataset:
            # Multi-task SC expanded by individual datasets
            label = f'<a href="#{section_anchor}">{sc}</a>'
            lines.append(
                f'<th>{label} '
                f'<button class="toggle-btn" onclick="toggleOvScLang(this,\'{sc_slug}\')">+</button></th>'
            )
            for ds in sc_datasets[sc]:
                metric = sc_dataset_metric[sc][ds]
                unit = " %" if metric in ZERO_TO_ONE_RANGE else ""
                lines.append(
                    f'<th class="lang-col" data-sc-lang="{sc_slug}">'
                    f'{ds} ({metric.upper()}{unit})</th>'
                )
        elif sc in expandable_lang:
            # Single-task SC with multiple languages: [+] for language sub-columns
            task = sc_tasks[sc][0]
            metric = task_metric[task]
            unit = " %" if metric in ZERO_TO_ONE_RANGE else ""
            label = f'<a href="#{section_anchor}">{sc}</a> ({metric.upper()}{unit})'
            lines.append(
                f'<th>{label} '
                f'<button class="toggle-btn" onclick="toggleOvScLang(this,\'{sc_slug}\')">+</button></th>'
            )
            for lang in task_languages[task]:
                ds_list = task_lang_datasets.get(task, {}).get(lang, [])
                header = f"{lang} - {ds_list[0]}" if len(ds_list) == 1 else lang
                lines.append(
                    f'<th class="lang-col" data-sc-lang="{sc_slug}">'
                    f'{header}</th>'
                )
        else:
            # Single-task SC: show task name directly with metric
            task = sc_tasks[sc][0]
            metric = task_metric[task]
            unit = " %" if metric in ZERO_TO_ONE_RANGE else ""
            label = f'<a href="#{section_anchor}">{sc}</a> ({metric.upper()}{unit})'
            lines.append(f"<th>{label}</th>")

    lines.append("</tr></thead>")

    # --- Body ---
    lines.append("<tbody>")
    for ri, m in enumerate(sorted_models):
        lines.append("<tr>")
        lines.append(f"<td>{m}</td>")
        lines.append(f"<td>{_extract_model_size(m)}</td>")

        # Aggregate columns
        for agg_name in table_aggregates:
            ar = agg_render[agg_name]
            v = ar["values"][m]
            if v != ar["sentinel"]:
                lines.append(_td(f"{v:{ar['fmt']}}", rank_key=ar["ranks"].get(ri)))
            else:
                lines.append(_td("-", is_missing=True))

        for sc in super_cats:
            sc_slug = _slug(sc)

            if sc in expandable_sc:
                # Aggregate SC cell
                if m in sc_model_score[sc]:
                    val = sc_model_score[sc][m]
                    lines.append(_td(f"{val:.2f}", rank_key=sc_ranks.get(sc, {}).get(ri),
                                     title=f"Avg across {', '.join(sc_tasks[sc])}"))
                else:
                    lines.append(_td("-", is_missing=True))

                # Per-task sub-cells (hidden)
                for task in sc_tasks[sc]:
                    metric = task_metric[task]
                    attr = f' class="lang-col" data-sc="{sc_slug}"'
                    if m in task_model_score[task]:
                        sc_val, st, n = task_model_score[task][m]
                        html_v, tip = _format_score_with_ci(sc_val, metric, st, n)
                        lines.append(_td(html_v, rank_key=task_ranks.get(task, {}).get(ri),
                                         extra_attrs=attr, title=tip))
                    else:
                        lines.append(_td("-", is_missing=True, extra_attrs=attr))
            elif sc in expandable_dataset:
                # Multi-task SC with per-dataset expansion
                if m in sc_model_score[sc]:
                    val = sc_model_score[sc][m]
                    lines.append(_td(f"{val:.2f}", rank_key=sc_ranks.get(sc, {}).get(ri),
                                     title=f"Avg across {len(sc_datasets[sc])} datasets"))
                else:
                    lines.append(_td("-", is_missing=True))

                # Per-dataset sub-cells (hidden)
                for ds in sc_datasets[sc]:
                    metric = sc_dataset_metric[sc][ds]
                    attr = f' class="lang-col" data-sc-lang="{sc_slug}"'
                    if m in sc_dataset_scores[sc][ds]:
                        ds_val, st, n = sc_dataset_scores[sc][ds][m]
                        html_v, tip = _format_score_with_ci(ds_val, metric, st, n)
                        lines.append(_td(html_v, rank_key=sc_dataset_ranks[sc].get(ds, {}).get(ri),
                                         extra_attrs=attr, title=tip))
                    else:
                        lines.append(_td("-", is_missing=True, extra_attrs=attr))
            elif sc in expandable_lang:
                # Single-task SC with language expansion
                task = sc_tasks[sc][0]
                metric = task_metric[task]
                if m in task_model_score[task]:
                    sc_val, st, n = task_model_score[task][m]
                    html_v, tip = _format_score_with_ci(sc_val, metric, st, n)
                    lines.append(_td(html_v, rank_key=task_ranks.get(task, {}).get(ri), title=tip))
                else:
                    lines.append(_td("-", is_missing=True))

                # Per-language sub-cells (hidden)
                for lang in task_languages[task]:
                    attr = f' class="lang-col" data-sc-lang="{sc_slug}"'
                    if m in task_lang_scores[task][lang]:
                        lv, st, n = task_lang_scores[task][lang][m]
                        html_v, tip = _format_score_with_ci(lv, metric, st, n)
                        lines.append(_td(html_v, rank_key=task_lang_ranks[task].get(lang, {}).get(ri),
                                         extra_attrs=attr, title=tip))
                    else:
                        lines.append(_td("-", is_missing=True, extra_attrs=attr))
            else:
                # Single-task SC: show task score directly
                task = sc_tasks[sc][0]
                metric = task_metric[task]
                if m in task_model_score[task]:
                    sc_val, st, n = task_model_score[task][m]
                    html_v, tip = _format_score_with_ci(sc_val, metric, st, n)
                    lines.append(_td(html_v, rank_key=task_ranks.get(task, {}).get(ri), title=tip))
                else:
                    lines.append(_td("-", is_missing=True))

        lines.append("</tr>")

    lines.append("</tbody></table>")

    # JavaScript toggle
    lines.append("""\
<script>
function toggleOvSc(btn, sc) {
  var tbl = btn.closest('table');
  var cells = tbl.querySelectorAll('[data-sc="' + sc + '"]');
  if (!cells.length) return;
  var show = cells[0].style.display !== 'table-cell';
  for (var i = 0; i < cells.length; i++)
    cells[i].style.display = show ? 'table-cell' : 'none';
  btn.textContent = show ? '\\u2212' : '+';
}
function toggleOvScLang(btn, sc) {
  var tbl = btn.closest('table');
  var cells = tbl.querySelectorAll('[data-sc-lang="' + sc + '"]');
  if (!cells.length) return;
  var show = cells[0].style.display !== 'table-cell';
  for (var i = 0; i < cells.length; i++)
    cells[i].style.display = show ? 'table-cell' : 'none';
  btn.textContent = show ? '\\u2212' : '+';
}
</script>""")

    collector.append({
        "category": title,
        "chart_type": "table",
        "metric": "overview",
        "raw_html": "\n".join(lines),
    })

    return data


# ---------------------------------------------------------------------------
# Plotting — Summary Tables (per-task, expandable per-dataset)
# ---------------------------------------------------------------------------

def plot_summary_tables(raw_entries, collector, category_override=None, subtitle=None,
                        table_aggregates=None):
    """Build summary HTML tables for all tasks with expandable per-dataset sub-columns.

    Each language column has a [+] toggle that reveals the individual dataset
    scores, and a clickable link to the corresponding per-dataset section.

    When sub_task grouping differs from language grouping, both views are
    rendered with a toggle bar above the tables.

    *category_override*: if set, use this as the collector category instead of
    "Tasks · {task}".  *subtitle*: if set, prepend a sub-header before each table.
    """
    for task, task_raw in sorted(_group_by_task(raw_entries).items()):
        if not task_raw:
            continue

        agg_lang = aggregate_entries(raw_entries, task_filter=task, by_language=True)
        if not agg_lang:
            continue

        agg_sub = aggregate_entries(raw_entries, task_filter=task, by_subtask=True)

        # Check if sub-task grouping differs from language grouping
        lang_keys = sorted({e["dataset_name"] for e in agg_lang}, key=_lang_sort_key)
        sub_keys = sorted({e["dataset_name"] for e in agg_sub}, key=_lang_sort_key)
        has_dual = len(lang_keys) >= 2 and len(sub_keys) >= 2 and lang_keys != sub_keys

        for metric in sorted({e["metric_name"] for e in agg_lang}):
            if has_dual:
                _build_dual_summary_tables(task, metric, task_raw,
                                           agg_lang, agg_sub, collector,
                                           category_override=category_override,
                                           subtitle=subtitle,
                                           table_aggregates=table_aggregates)
            else:
                _build_summary_table(task, metric, task_raw, agg_lang,
                                     collector, group_key_fn=None,
                                     category_override=category_override,
                                     subtitle=subtitle,
                                     table_aggregates=table_aggregates)


def _build_summary_table(task, metric, task_raw, agg_lang, collector,
                         group_key_fn=None, tbl_id_suffix="",
                         category_override=None, subtitle=None,
                         table_aggregates=None):
    """Build one HTML summary table for a (task, metric) pair.

    *group_key_fn*, when provided, maps a raw entry to its group label
    (used for sub-task grouping).  When None, groups by language.
    *tbl_id_suffix* is appended to the table HTML id for uniqueness.
    *category_override*: if set, use as collector category instead of "Tasks · {task}".
    *subtitle*: if set, prepend a sub-header before the table title.
    """
    agg_metric = [e for e in agg_lang if e["metric_name"] == metric]
    raw_metric = [e for e in task_raw if e["metric_name"] == metric]
    if not agg_metric:
        return

    all_models = sorted({e["model_name"] for e in agg_metric})
    languages = sorted({e["dataset_name"] for e in agg_metric}, key=_lang_sort_key)  # dataset_name = group key

    # lang -> model -> (score, std, n)
    lang_model_score = defaultdict(dict)
    for e in agg_metric:
        lang_model_score[e["dataset_name"]][e["model_name"]] = (
            e["score"], e.get("std"), e.get("n")
        )

    # Per-dataset breakdown: group -> dataset -> model -> (score, std, n)
    lang_ds_model = defaultdict(lambda: defaultdict(dict))
    lang_datasets = defaultdict(set)
    for e in raw_metric:
        lang = group_key_fn(e) if group_key_fn else (e["language"] or "UNKNOWN").upper()
        ds_display = _dataset_display_name(e)
        lang_ds_model[lang][ds_display][e["model_name"]] = (
            e["score"], e.get("std"), e.get("n")
        )
        lang_datasets[lang].add(ds_display)
    lang_datasets = {l: sorted(ds) for l, ds in lang_datasets.items()}

    expandable_langs = {l for l in languages if len(lang_datasets.get(l, [])) >= 2}

    # Average per model across languages
    ascending = _sort_ascending(metric)
    # Compute model_avg (needed for the "Average" column)
    model_avg = {}
    for m in all_models:
        scores = [lang_model_score[l][m][0] for l in languages if m in lang_model_score[l]]
        model_avg[m] = sum(scores) / len(scores) if scores else None

    # Per-language ranks and avg rank per model
    lang_model_rank = {}
    for lang in languages:
        ranked = sorted(
            [(lang_model_score[lang][m][0], m) for m in all_models if m in lang_model_score[lang]],
            key=lambda x: x[0], reverse=not ascending,
        )
        lang_model_rank[lang] = {m: rank + 1 for rank, (_, m) in enumerate(ranked)}

    model_avg_rank = {}
    for m in all_models:
        ranks = [lang_model_rank[l][m] for l in languages if m in lang_model_rank.get(l, {})]
        model_avg_rank[m] = sum(ranks) / len(ranks) if ranks else float("inf")

    # Normalized scores: build per-language display-score dicts
    lang_disp_scores = {}
    for lang in languages:
        lang_disp_scores[lang] = {
            m: _display_score(lang_model_score[lang][m][0], metric)
            for m in lang_model_score[lang]
        }
    model_minmax, model_zscore = _compute_normalized_scores(
        all_models, lang_disp_scores, {lang: ascending for lang in languages},
    )

    # Sort models by the first requested aggregate
    agg_values = {"avg_rank": model_avg_rank, "minmax": model_minmax, "zscore": model_zscore}
    sorted_models = _sort_models_by_aggregate(all_models, agg_values, table_aggregates[0])

    # --- Ranked row indices for highlighting ---
    lang_ranks = {}
    for lang in languages:
        pairs = [
            (_display_score(lang_model_score[lang][m][0], metric), ri)
            for ri, m in enumerate(sorted_models) if m in lang_model_score[lang]
        ]
        lang_ranks[lang] = _ranked_rows(pairs, ascending)

    lang_ds_ranks = defaultdict(dict)
    for lang in expandable_langs:
        for ds in lang_datasets.get(lang, []):
            pairs = [
                (_display_score(lang_ds_model[lang][ds][m][0], metric), ri)
                for ri, m in enumerate(sorted_models) if m in lang_ds_model[lang][ds]
            ]
            lang_ds_ranks[lang][ds] = _ranked_rows(pairs, ascending)

    avg_ranks = _ranked_rows(
        [(_display_score(model_avg[m], metric), ri)
         for ri, m in enumerate(sorted_models) if model_avg[m] is not None],
        ascending,
    )

    agg_render = _agg_columns_html(table_aggregates, sorted_models, agg_values)

    # --- Build HTML ---
    cat_name = category_override or ("Tasks \u00b7 " + task)
    tbl_id = "sum-" + _slug(task) + "-" + _slug(metric) + tbl_id_suffix

    lines = []

    if subtitle:
        lines.append(
            f'<div style="font-size:14px;font-weight:600;color:#1e293b;margin:14px 0 4px;'
            f'border-left:3px solid #3b82f6;padding-left:8px">{subtitle}</div>'
        )
    title = _format_suptitle(cat_name, metric)
    lines.append(
        f'<div style="font-size:15px;font-weight:600;color:#475569;margin:8px 0">{title}</div>'
    )

    lines.append(f'<table class="ov-tbl" id="{tbl_id}">')

    # Header
    lines.append("<thead><tr>")
    lines.append("<th>Model</th>")
    lines.append("<th>Size</th>")
    for agg_name in table_aggregates:
        lines.append(f"<th>{_AGG_META[agg_name]['label']}</th>")
    lines.append("<th>Average</th>")
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
            ds_list = lang_datasets.get(lang, [])
            if len(ds_list) == 1:
                lines.append(f"<th>{lang} - {ds_list[0]}</th>")
            else:
                lines.append(f"<th>{lang}</th>")
    lines.append("</tr></thead>")

    # Body
    lines.append("<tbody>")
    for ri, m in enumerate(sorted_models):
        lines.append("<tr>")
        lines.append(f"<td>{m}</td>")
        lines.append(f"<td>{_extract_model_size(m)}</td>")

        # Aggregate columns
        for agg_name in table_aggregates:
            ar = agg_render[agg_name]
            v = ar["values"][m]
            if v != ar["sentinel"]:
                lines.append(_td(f"{v:{ar['fmt']}}", rank_key=ar["ranks"].get(ri)))
            else:
                lines.append(_td("-", is_missing=True))

        # Average
        if model_avg[m] is not None:
            val = f"{_display_score(model_avg[m], metric):.2f}"
            lines.append(_td(val, rank_key=avg_ranks.get(ri)))
        else:
            lines.append(_td("-", is_missing=True))

        for lang in languages:
            grp = _slug(lang)

            # Aggregate cell
            if m in lang_model_score[lang]:
                sc, st, n = lang_model_score[lang][m]
                html_v, tip = _format_score_with_ci(sc, metric, st, n)
                lines.append(_td(html_v, rank_key=lang_ranks.get(lang, {}).get(ri), title=tip))
            else:
                lines.append(_td("-", is_missing=True))

            # Per-dataset sub-cells
            if lang in expandable_langs:
                for ds in lang_datasets[lang]:
                    attr = f' class="lang-col" data-group="{grp}"'
                    if m in lang_ds_model[lang][ds]:
                        sc, st, n = lang_ds_model[lang][ds][m]
                        html_v, tip = _format_score_with_ci(sc, metric, st, n)
                        lines.append(_td(html_v, rank_key=lang_ds_ranks[lang].get(ds, {}).get(ri),
                                         extra_attrs=attr, title=tip))
                    else:
                        lines.append(_td("-", is_missing=True, extra_attrs=attr))

        lines.append("</tr>")

    lines.append("</tbody></table>")

    # JS — generic toggle (harmless if redefined by other tables)
    lines.append(_TOGGLE_COLS_JS)

    collector.append({
        "category": cat_name,
        "chart_type": "table",
        "metric": metric,
        "raw_html": "\n".join(lines),
    })


def _build_dual_summary_tables(task, metric, task_raw, agg_lang, agg_sub,
                                collector, category_override=None, subtitle=None,
                                table_aggregates=None):
    """Build two summary tables (language & sub-task) with a toggle bar."""
    # Collect HTML from each view into a temporary list
    lang_collector = []
    _build_summary_table(task, metric, task_raw, agg_lang, lang_collector,
                         group_key_fn=None, tbl_id_suffix="-lang",
                         category_override=category_override, subtitle=subtitle,
                         table_aggregates=table_aggregates)
    sub_collector = []
    _build_summary_table(task, metric, task_raw, agg_sub, sub_collector,
                         group_key_fn=_effective_subtask, tbl_id_suffix="-sub",
                         category_override=category_override, subtitle=subtitle,
                         table_aggregates=table_aggregates)

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

    cat_name = category_override or ("Tasks \u00b7 " + task)
    collector.append({
        "category": cat_name,
        "chart_type": "table",
        "metric": metric,
        "raw_html": "\n".join(lines),
    })


# ---------------------------------------------------------------------------
# Plotting — Language Sections
# ---------------------------------------------------------------------------

def plot_language_sections(entries, collector, include_violin=False,
                           table_aggregates=None):
    """Build per-language-group sections (French, English, Others).

    Each group gets violin plots per task (when *include_violin* is True)
    and a summary table (Models × Tasks) with expandable per-dataset
    sub-columns.
    """
    # Classify entries by language group
    group_entries = defaultdict(list)
    for e in entries:
        grp = _classify_language(e.get("language"))
        group_entries[grp].append(e)

    for group_name in ["French", "English", "Others"]:
        grp_ents = group_entries.get(group_name, [])
        if not grp_ents:
            continue

        category = f"Languages \u00b7 {group_name}"

        # Violin plots per task
        task_raw_map = defaultdict(list)
        for e in grp_ents:
            task = e.get("task")
            if task:
                task_raw_map[task].append(e)

        if include_violin:
            for task in sorted(task_raw_map.keys()):
                if task_raw_map[task]:
                    plot_violin_charts(task_raw_map[task], category, collector)

        # Summary table: Models × Tasks
        _build_language_summary_table(grp_ents, group_name, category, collector,
                                       table_aggregates=table_aggregates)


def _build_language_summary_table(entries, lang_group, category, collector,
                                   table_aggregates=None):
    """Build a summary table for a language group: Models × Tasks with expandable datasets."""
    # Group entries by task
    task_entries_map = defaultdict(list)
    for e in entries:
        task = e.get("task")
        if task:
            task_entries_map[task].append(e)

    if not task_entries_map:
        return

    tasks = sorted(task_entries_map.keys())
    all_models = sorted({e["model_name"] for e in entries})

    # Pick most common metric per task
    task_metric = {}
    for task, ents in task_entries_map.items():
        override = _TASK_METRIC_OVERRIDE.get(task.upper())
        if override and any(e["metric_name"] == override for e in ents):
            task_metric[task] = override
        else:
            task_metric[task] = _most_common_metric(ents)

    # task -> model -> (avg_score, std, n)
    task_model_score = {}
    # task -> dataset -> model -> (score, std, n)
    task_ds_model = defaultdict(lambda: defaultdict(dict))
    task_datasets = defaultdict(set)

    for task in tasks:
        metric = task_metric[task]
        model_scores = {}
        for m in all_models:
            matching = [
                e for e in task_entries_map[task]
                if e["model_name"] == m and e["metric_name"] == metric
            ]
            scores = [e["score"] for e in matching]
            if scores:
                pooled = []
                for e in matching:
                    if "all_scores" in e:
                        pooled.extend(e["all_scores"])
                avg = sum(scores) / len(scores)
                if pooled:
                    model_scores[m] = (avg, float(np.std(np.array(pooled))), len(pooled))
                else:
                    model_scores[m] = (avg, None, None)
            # Per-dataset breakdown
            for e in matching:
                ds_display = _dataset_display_name(e)
                task_ds_model[task][ds_display][m] = (
                    e["score"], e.get("std"), e.get("n")
                )
                task_datasets[task].add(ds_display)
        task_model_score[task] = model_scores

    task_datasets = {t: sorted(ds) for t, ds in task_datasets.items()}
    expandable_tasks = {t for t in tasks if len(task_datasets.get(t, [])) >= 1}

    # Compute aggregate scores
    ascending_map = {t: _sort_ascending(task_metric[t]) for t in tasks}
    task_model_rank = {}
    for task in tasks:
        asc = ascending_map[task]
        scores = task_model_score[task]
        ranked = sorted(scores.items(), key=lambda x: x[1][0], reverse=not asc)
        task_model_rank[task] = {m: rank + 1 for rank, (m, _) in enumerate(ranked)}

    model_avg_rank = {}
    for m in all_models:
        ranks = [task_model_rank[t][m] for t in tasks if m in task_model_rank[t]]
        model_avg_rank[m] = sum(ranks) / len(ranks) if ranks else float("inf")

    # Normalized scores: build per-task display-score dicts
    task_disp_scores = {}
    for task in tasks:
        metric = task_metric[task]
        task_disp_scores[task] = {
            m: _display_score(task_model_score[task][m][0], metric)
            for m in task_model_score[task]
        }
    model_minmax, model_zscore = _compute_normalized_scores(
        all_models, task_disp_scores, ascending_map,
    )

    # Sort models by the first requested aggregate
    agg_values = {"avg_rank": model_avg_rank, "minmax": model_minmax, "zscore": model_zscore}
    sorted_models = _sort_models_by_aggregate(all_models, agg_values, table_aggregates[0])

    # Ranked rows for highlighting
    task_ranks = {}
    for task in tasks:
        metric = task_metric[task]
        asc = ascending_map[task]
        pairs = [
            (_display_score(task_model_score[task][m][0], metric), ri)
            for ri, m in enumerate(sorted_models) if m in task_model_score[task]
        ]
        task_ranks[task] = _ranked_rows(pairs, asc)

    task_ds_ranks = defaultdict(dict)
    for task in expandable_tasks:
        metric = task_metric[task]
        asc = ascending_map[task]
        for ds in task_datasets[task]:
            pairs = [
                (_display_score(task_ds_model[task][ds][m][0], metric), ri)
                for ri, m in enumerate(sorted_models) if m in task_ds_model[task][ds]
            ]
            task_ds_ranks[task][ds] = _ranked_rows(pairs, asc)

    agg_render = _agg_columns_html(table_aggregates, sorted_models, agg_values)

    # --- Build HTML ---
    tbl_id = "lang-" + _slug(lang_group)

    lines = []
    lines.append(
        f'<div style="font-size:15px;font-weight:600;color:#475569;margin:8px 0">'
        f'{lang_group} — Models \u00d7 Tasks</div>'
    )

    lines.append(f'<table class="ov-tbl" id="{tbl_id}">')

    # Header
    lines.append("<thead><tr>")
    lines.append("<th>Model</th>")
    lines.append("<th>Size</th>")
    for agg_name in table_aggregates:
        lines.append(f"<th>{_AGG_META[agg_name]['label']}</th>")
    for task in tasks:
        metric = task_metric[task]
        unit = " %" if metric in ZERO_TO_ONE_RANGE else ""
        slug = _slug(task)
        label = f'{task} ({metric.upper()}{unit})'
        if task in expandable_tasks:
            lines.append(
                f'<th>{label} '
                f'<button class="toggle-btn" onclick="toggleCols(this,\'{tbl_id}\',\'{slug}\')">+</button></th>'
            )
            for ds in task_datasets[task]:
                lines.append(f'<th class="lang-col" data-group="{slug}">{ds}</th>')
        else:
            lines.append(f"<th>{label}</th>")
    lines.append("</tr></thead>")

    # Body
    lines.append("<tbody>")
    for ri, m in enumerate(sorted_models):
        # Skip models with no data in this group
        if not any(m in task_model_score[t] for t in tasks):
            continue
        lines.append("<tr>")
        lines.append(f"<td>{m}</td>")
        lines.append(f"<td>{_extract_model_size(m)}</td>")

        # Aggregate columns
        for agg_name in table_aggregates:
            ar = agg_render[agg_name]
            v = ar["values"][m]
            if v != ar["sentinel"]:
                lines.append(_td(f"{v:{ar['fmt']}}", rank_key=ar["ranks"].get(ri)))
            else:
                lines.append(_td("-", is_missing=True))

        for task in tasks:
            metric = task_metric[task]
            slug = _slug(task)

            if m in task_model_score[task]:
                sc, st, n = task_model_score[task][m]
                html_v, tip = _format_score_with_ci(sc, metric, st, n)
                lines.append(_td(html_v, rank_key=task_ranks.get(task, {}).get(ri), title=tip))
            else:
                lines.append(_td("-", is_missing=True))

            if task in expandable_tasks:
                for ds in task_datasets[task]:
                    attr = f' class="lang-col" data-group="{slug}"'
                    if m in task_ds_model[task][ds]:
                        sc, st, n = task_ds_model[task][ds][m]
                        html_v, tip = _format_score_with_ci(sc, metric, st, n)
                        lines.append(_td(html_v, rank_key=task_ds_ranks[task].get(ds, {}).get(ri),
                                         extra_attrs=attr, title=tip))
                    else:
                        lines.append(_td("-", is_missing=True, extra_attrs=attr))

        lines.append("</tr>")

    lines.append("</tbody></table>")

    # JS toggle (harmless if redefined)
    lines.append(_TOGGLE_COLS_JS)

    collector.append({
        "category": category,
        "chart_type": "table",
        "metric": "overview",
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
    within each group.  The sidebar is split into **Overview**, **Tasks**,
    and **Languages** groups.
    """
    from collections import OrderedDict

    # Group figures by category, preserving insertion order
    categories = OrderedDict()
    for item in collected_figures:
        cat = item["category"]
        categories.setdefault(cat, []).append(item)

    # --- Classify categories into overview, tasks, languages ---
    _TASKS_PREFIX = "Tasks \u00b7 "
    _LANG_PREFIX = "Languages \u00b7 "
    overview_cats = []          # category_name
    tasks_cats = []             # (task_label, category_name)
    lang_cats = []              # (lang_label, category_name)

    for cat in categories:
        if cat == "Overview" or cat.startswith("Overview"):
            overview_cats.append(cat)
        elif cat.startswith(_TASKS_PREFIX):
            task = cat[len(_TASKS_PREFIX):]
            tasks_cats.append((task, cat))
        elif cat.startswith(_LANG_PREFIX):
            lang = cat[len(_LANG_PREFIX):]
            lang_cats.append((lang, cat))

    # --- Build nav HTML ---
    nav_lines = []

    if overview_cats:
        nav_lines.append('    <li class="nav-group">Overview</li>')
        for cat in overview_cats:
            slug = _slug(cat)
            label = "All Tasks" if cat == "Overview" else cat.replace("Overview ", "")
            nav_lines.append(f'    <li><a href="#cat-{slug}">{label}</a></li>')

    if tasks_cats:
        nav_lines.append('    <li class="nav-group">Tasks</li>')
        for task, cat in tasks_cats:
            slug = _slug(cat)
            nav_lines.append(f'    <li><a href="#cat-{slug}">{task}</a></li>')

    if lang_cats:
        nav_lines.append('    <li class="nav-group">Languages</li>')
        for lang, cat in lang_cats:
            slug = _slug(cat)
            nav_lines.append(f'    <li><a href="#cat-{slug}">{lang}</a></li>')

    # --- Build section HTML ---
    section_blocks = []
    fig_counter = 0

    for cat, items in categories.items():
        slug = _slug(cat)

        violins = [it for it in items if it["chart_type"] == "violin"]
        tables = [it for it in items if it["chart_type"] == "table"]

        section_html = f'<section class="category" id="cat-{slug}">\n  <h2>{cat}</h2>\n'

        for chart_label, chart_items in [("Tables", tables), ("Score Distributions", violins)]:
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
        description="Plot AudioBench evaluation results as a single interactive HTML report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_folder", help="Path to results folder (e.g. results/)")
    parser.add_argument("--output_folder", type=str, default="plots/", help="Where to save report")
    parser.add_argument("--violin", action="store_true", help="Include violin plots in the report")
    parser.add_argument(
        "--table_aggregates", nargs="+", default=AGGREGATE_MEASURES,
        choices=AGGREGATE_MEASURES,
        help="Aggregate columns to show in the overview table",
    )
    parser.add_argument(
        "--figure_aggregates", nargs="+", default=["minmax"],
        choices=AGGREGATE_MEASURES,
        help="Aggregate measure(s) for the size-vs-performance figure(s)",
    )
    args = parser.parse_args()

    # Load all scores
    entries = load_all_scores(args.input_folder)
    if not entries:
        print(f"No score files found in {args.input_folder}")
        return

    # Filter out Arabic entries globally
    entries = [e for e in entries if (e.get("language") or "").upper() != "AR"]

    print(f"Loaded {len(entries)} score entries from {len({e['model_name'] for e in entries})} models")

    collector = []

    # --- Step 0: Overview table (all tasks × models) ---
    overview_data = plot_overview_table(entries, collector,
                                        table_aggregates=args.table_aggregates)
    plot_size_vs_performance(entries, collector, overview_data=overview_data,
                             figure_aggregates=args.figure_aggregates)

    # --- Step 0b: Filtered overview (FR/EN, ASR/AST/QA only) ---
    # For AST, include entries where source or target language is FR or EN
    # (language field can be e.g. "fr-en", "es-fr", "en")
    _allowed_sc = {"ASR", "AST", "QA"}
    _fren = {"FR", "EN"}
    def _lang_match(entry):
        lang = (entry.get("language") or "").upper()
        parts = lang.split("-")
        return any(p in _fren for p in parts)

    filtered = [
        e for e in entries
        if _lang_match(e)
        and _super_category(e.get("task", "")) in _allowed_sc
    ]
    if filtered:
        filtered_data = plot_overview_table(
            filtered, collector,
            title="Overview (FR/EN \u2014 ASR, AST, QA)",
            table_id="overview-filtered-tbl",
            allowed_super_cats=_allowed_sc,
            table_aggregates=args.table_aggregates,
        )
        plot_size_vs_performance(
            filtered, collector,
            category="Overview (FR/EN \u2014 ASR, AST, QA)",
            overview_data=filtered_data,
            figure_aggregates=args.figure_aggregates,
        )

    # --- Steps 1+2: Super-category sections (violin plots + summary tables) ---
    for super_cat, task_map in _group_by_super_category(entries).items():
        cat_label = f"Tasks \u00b7 {super_cat}"
        multi_task = len(task_map) > 1

        # Violin plots: one per task within the super-category
        if args.violin:
            for task, task_raw in sorted(task_map.items()):
                if task_raw:
                    plot_violin_charts(task_raw, cat_label, collector)

        # Summary tables per task within the super-category
        for task, task_raw in sorted(task_map.items()):
            subtitle = task if multi_task else None
            plot_summary_tables(task_raw, collector,
                                category_override=cat_label,
                                subtitle=subtitle,
                                table_aggregates=args.table_aggregates)

    # --- Step 3: Language sections (French, English, Others) ---
    plot_language_sections(entries, collector, include_violin=args.violin,
                           table_aggregates=args.table_aggregates)

    if not collector:
        print("No figures generated.")
        return

    output_path = os.path.join(args.output_folder, "report.html")
    build_html_report(collector, output_path)


if __name__ == "__main__":
    main()
