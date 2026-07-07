#!/usr/bin/env python3
"""Pairwise statistical comparison of two AudioBench systems.

Unlike ``plot_results_to_html.py`` (which prints raw scores for every system on
every dataset), this script compares exactly **two** systems sample-by-sample
and runs paired statistical tests to decide whether one is significantly better
than the other.

For every ``*_score.json`` shared by both systems it pairs the per-sample scores
(``all_scores``), pools them per ``{task, language, metric}`` group, and runs:

* a **paired Student's t-test** (``scipy.stats.ttest_rel``), and
* a **Wilcoxon signed-rank test** (``scipy.stats.wilcoxon``) — robust to the
  heavy outliers found in WER / judge metrics.

Results are written as a single self-contained HTML report containing:

* a per-``{task, language, metric}`` summary table,
* a forest plot of the standardized effect size (Cohen's dz) with 95% CI,
* per-metric grouped-bar charts of the two systems' means with 95% CI,
* a per-dataset breakdown table.

Safety checks (the systems must be compared on *exactly* the same data):

* a **warning** is printed when a ``*_score.json`` exists for one system but not
  the other,
* the script **fails** when a shared file has a different number of scores, and
* by default it also verifies that the reference texts line up in the same order
  (read from the companion ``<dataset>.json`` files). Disable with
  ``--dont-check-ref``.

Usage:
    python audio_bench/compare_two_systems.py results/ SYSTEM_A SYSTEM_B
    python audio_bench/compare_two_systems.py results/ SYSTEM_A SYSTEM_B \\
        --output_folder plots/ --alpha 0.05 --dont-check-ref

SYSTEM_A / SYSTEM_B are the sub-folder names under the results folder
(e.g. ``linagora_xp_data_v3_canary_rote``), or absolute/relative paths to them.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Reuse the metric conventions (direction + %-scaling) from the plotting script
# so the two tools never disagree about how a metric is interpreted.
try:
    from visualization.plot_results_to_html import (  # when run as a script from audio_bench/
        LOWER_IS_BETTER,
        ZERO_TO_ONE_RANGE,
        _TASK_METRIC_OVERRIDE,
        _lang_sort_key,
        _task_display_name,
    )
except ImportError:  # when run as a module: python -m audio_bench.compare_two_systems
    from audio_bench.visualization.plot_results_to_html import (
        LOWER_IS_BETTER,
        ZERO_TO_ONE_RANGE,
        _TASK_METRIC_OVERRIDE,
        _lang_sort_key,
        _task_display_name,
    )

# Metrics that are unbounded above and dominated by a few catastrophic samples
# (e.g. WER can exceed 100% when the model loops/hallucinates). Per-sample values
# are clipped to this ceiling before any statistic/test so that means and tests
# reflect typical behaviour rather than a handful of outliers.
CLIP_AT_ONE = {"wer"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def resolve_system_dir(results_folder, name):
    """Resolve *name* to a system directory, as a path or a sub-folder name."""
    p = Path(name)
    if p.is_dir():
        return p
    p = Path(results_folder) / name
    if p.is_dir():
        return p
    sys.exit(f"Error: system directory not found: '{name}' "
             f"(looked for '{name}' and '{Path(results_folder) / name}')")


def collect_score_files(system_dir):
    """Return {relative_path: Path} for every ``*_score.json`` under *system_dir*."""
    files = {}
    for path in sorted(system_dir.rglob("*_score.json")):
        files[str(path.relative_to(system_dir))] = path
    return files


def load_metric_scores(score_path, metric):
    """Return the per-sample ``all_scores`` list for *metric*, or None."""
    try:
        data = json.loads(score_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        sys.exit(f"Error: cannot read {score_path}: {exc}")
    raw = data.get(metric)
    if not isinstance(raw, dict):
        return None
    scores = raw.get("all_scores")
    if not isinstance(scores, list):
        return None
    return scores


def load_score_meta(score_path):
    """Return (metrics, task, language) for a score file."""
    data = json.loads(score_path.read_text())
    return data.get("metrics", []), data.get("task"), data.get("language")


def load_references(score_path):
    """Read reference strings (in order) from the companion ``<dataset>.json``.

    Returns the list of references, or None if the companion file is missing or
    has no usable references.
    """
    companion = score_path.with_name(score_path.name.replace("_score.json", ".json"))
    if not companion.exists():
        return None
    try:
        data = json.loads(companion.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    predictions = data.get("predictions")
    if not isinstance(predictions, list):
        return None
    return [p.get("reference") for p in predictions]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _sig_stars(p, alpha):
    if p is None or math.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns" if p >= alpha else "*"


def paired_stats(a, b, metric, alpha):
    """Run paired tests on two equal-length per-sample score lists.

    Returns a dict of statistics.  Differences are *oriented* so that a positive
    advantage always means system A is better, taking each metric's
    maximize/minimize direction into account.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    lower_better = metric in LOWER_IS_BETTER

    # Oriented per-sample advantage of A (positive => A better)
    d = (b - a) if lower_better else (a - b)
    mean_d = float(np.mean(d)) if n else float("nan")

    sf = 100.0 if metric in ZERO_TO_ONE_RANGE else 1.0
    mean_a = float(np.mean(a)) * sf if n else float("nan")
    mean_b = float(np.mean(b)) * sf if n else float("nan")
    median_a = float(np.median(a)) * sf if n else float("nan")
    median_b = float(np.median(b)) * sf if n else float("nan")

    # Two-sided p-values
    t_p = wil_p = float("nan")
    if n >= 2 and np.any(a != b):
        try:
            t_p = float(stats.ttest_rel(a, b).pvalue)
        except (ValueError, FloatingPointError):
            t_p = float("nan")
        try:
            wil_p = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            wil_p = float("nan")
    elif n >= 2:  # identical scores everywhere
        t_p = wil_p = 1.0

    # Standardized effect size (Cohen's dz) and its 95% CI
    sd = float(np.std(d, ddof=1)) if n >= 2 else 0.0
    if sd > 0:
        dz = mean_d / sd
        t_crit = float(stats.t.ppf(0.975, n - 1))
        ci_half = t_crit / math.sqrt(n)
        dz_lo, dz_hi = dz - ci_half, dz + ci_half
    else:
        dz = 0.0
        dz_lo = dz_hi = 0.0

    # Winner (by mean, respecting direction). Tie if no meaningful difference.
    if not math.isfinite(mean_d) or abs(mean_d) < 1e-12:
        winner = "tie"
    else:
        winner = "A" if mean_d > 0 else "B"

    # Display-scaled advantage of A and relative improvement of the winner
    adv_a = mean_d * sf
    base = mean_b if winner == "A" else mean_a
    rel_pct = (abs(adv_a) / abs(base) * 100.0) if base not in (0, float("nan")) and base else float("nan")

    # Do the two tests disagree about significance at this alpha?
    t_sig = (t_p is not None) and (not math.isnan(t_p)) and (t_p < alpha)
    w_sig = (wil_p is not None) and (not math.isnan(wil_p)) and (wil_p < alpha)
    disagree = (t_sig != w_sig)

    return {
        "n": n,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "median_a": median_a,
        "median_b": median_b,
        "adv_a": adv_a,          # display units, +ve => A better
        "rel_pct": rel_pct,      # winner improvement over loser, %
        "t_p": t_p,
        "wil_p": wil_p,
        "dz": dz,
        "dz_lo": dz_lo,
        "dz_hi": dz_hi,
        "winner": winner,
        "sig": _sig_stars(t_p, alpha),
        "disagree": disagree,
        "lower_better": lower_better,
    }


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def _clip(scores, metric, clip_wer):
    """Clip unbounded metrics to a sensible ceiling. Returns (scores, n_clipped)."""
    if not (clip_wer and metric in CLIP_AT_ONE):
        return scores, 0
    n_clipped = sum(1 for s in scores if s > 1.0)
    return [min(s, 1.0) for s in scores], n_clipped


def compare(results_folder, name_a, name_b, check_ref=True, clip_wer=True):
    """Pair up per-sample scores for the two systems.

    Returns (group_samples, dataset_rows, warnings) where:

    * group_samples maps (task, language, metric) -> {"a": [...], "b": [...]}
      with pooled per-sample scores,
    * dataset_rows is a list of per-dataset dicts (relpath, task, language,
      metric, a-scores, b-scores) for the detail table,
    * warnings is a list of human-readable warning strings.

    When *clip_wer* is True, per-sample WER values above 100% are clipped to
    100% before pooling so that means and tests are not dominated by a few
    catastrophic samples.
    """
    dir_a = resolve_system_dir(results_folder, name_a)
    dir_b = resolve_system_dir(results_folder, name_b)

    files_a = collect_score_files(dir_a)
    files_b = collect_score_files(dir_b)

    warnings = []
    clipped_total = 0
    for rel in sorted(set(files_a) - set(files_b)):
        warnings.append(f"'{rel}' present for {name_a} but missing for {name_b} — skipped")
    for rel in sorted(set(files_b) - set(files_a)):
        warnings.append(f"'{rel}' present for {name_b} but missing for {name_a} — skipped")

    common = sorted(set(files_a) & set(files_b))
    group_samples = defaultdict(lambda: {"a": [], "b": []})
    dataset_rows = []

    for rel in common:
        pa, pb = files_a[rel], files_b[rel]
        metrics_a, task_a, lang_a = load_score_meta(pa)
        metrics_b, task_b, lang_b = load_score_meta(pb)

        task = task_a or task_b
        language = lang_a or lang_b
        if task_a != task_b:
            warnings.append(f"'{rel}': task differs ({task_a} vs {task_b}); using '{task}'")
        if lang_a != lang_b:
            warnings.append(f"'{rel}': language differs ({lang_a} vs {lang_b}); using '{language}'")

        shared_metrics = [m for m in metrics_a if m in metrics_b]
        for m in metrics_a:
            if m not in metrics_b:
                warnings.append(f"'{rel}': metric '{m}' missing for {name_b} — skipped")

        # Reference-order check (once per file, independent of metric)
        if check_ref:
            refs_a = load_references(pa)
            refs_b = load_references(pb)
            if refs_a is None or refs_b is None:
                warnings.append(
                    f"'{rel}': could not load references for the order check "
                    f"(missing/invalid companion .json); skipping ref check for this file"
                )
            elif refs_a != refs_b:
                # Find first mismatch for a helpful message
                idx = next((i for i, (x, y) in enumerate(zip(refs_a, refs_b)) if x != y),
                           min(len(refs_a), len(refs_b)))
                sys.exit(
                    f"FAIL: references differ for '{rel}' at index {idx}.\n"
                    f"  {name_a}: {refs_a[idx] if idx < len(refs_a) else '<out of range>'!r}\n"
                    f"  {name_b}: {refs_b[idx] if idx < len(refs_b) else '<out of range>'!r}\n"
                    f"  (lengths: {len(refs_a)} vs {len(refs_b)})\n"
                    f"  The two systems were not evaluated on the same data in the same order.\n"
                    f"  Re-run with --dont-check-ref to bypass this check."
                )

        for m in shared_metrics:
            scores_a = load_metric_scores(pa, m)
            scores_b = load_metric_scores(pb, m)
            if scores_a is None or scores_b is None:
                warnings.append(f"'{rel}': metric '{m}' has no per-sample scores — skipped")
                continue
            if len(scores_a) != len(scores_b):
                sys.exit(
                    f"FAIL: '{rel}' metric '{m}' has a different number of scores "
                    f"({len(scores_a)} for {name_a} vs {len(scores_b)} for {name_b}).\n"
                    f"  The two systems were not evaluated on the same number of samples."
                )

            scores_a, ca = _clip(scores_a, m, clip_wer)
            scores_b, cb = _clip(scores_b, m, clip_wer)
            clipped_total += ca + cb

            key = (task, language, m)
            group_samples[key]["a"].extend(scores_a)
            group_samples[key]["b"].extend(scores_b)
            dataset_rows.append({
                "rel": rel,
                "dataset": Path(rel).name.removesuffix("_score.json"),
                "task": task,
                "language": language,
                "metric": m,
                "a": scores_a,
                "b": scores_b,
            })

    if clipped_total:
        warnings.append(
            f"Clipped {clipped_total} per-sample WER value(s) above 100% to 100% "
            f"(metrics {sorted(CLIP_AT_ONE)}); disable with --no-clip-wer"
        )

    return dict(group_samples), dataset_rows, warnings


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_FIRST_COLOR = "#5dade2"   # A better
_SECOND_COLOR = "#e67e22"  # B better
_NS_COLOR = "#95a5a6"      # not significant


def _fmt_p(p):
    if p is None or math.isnan(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _group_sort_key(key):
    task, lang, metric = key
    return (_task_display_name(task or ""), _lang_sort_key(lang or ""), metric)


def _winner_cell(s, name_a, name_b, alpha):
    """Return the HTML <td> for the winner column (color + name + stars + flags)."""
    winner_name = {"A": name_a, "B": name_b, "tie": "—"}[s["winner"]]
    if s["winner"] == "tie":
        color = _NS_COLOR
    elif s["t_p"] is not None and not math.isnan(s["t_p"]) and s["t_p"] < alpha:
        color = _FIRST_COLOR if s["winner"] == "A" else _SECOND_COLOR
    else:
        color = _NS_COLOR
    # Mark when the two tests disagree about significance at this alpha.
    flag = (' <span title="t-test and Wilcoxon disagree on significance '
            'at this &alpha; — treat with caution">&Dagger;</span>') if s.get("disagree") else ""
    return (f"<td style='background:{color};color:white;font-weight:600'>"
            f"{winner_name} {s['sig']}{flag}</td>")


def build_summary_table(group_stats, name_a, name_b, alpha):
    """HTML table, one row per {task, language, metric}."""
    rows = []
    for key in sorted(group_stats, key=_group_sort_key):
        task, lang, metric = key
        s = group_stats[key]
        unit = " %" if metric in ZERO_TO_ONE_RANGE else ""
        rel = f"{s['rel_pct']:.1f}%" if math.isfinite(s["rel_pct"]) else "—"
        rows.append(
            "<tr>"
            f"<td style='text-align:left'>{_task_display_name(task or '?')}</td>"
            f"<td>{(lang or '?')}</td>"
            f"<td>{metric}{unit}</td>"
            f"<td>{s['n']}</td>"
            f"<td>{s['mean_a']:.2f}</td>"
            f"<td>{s['mean_b']:.2f}</td>"
            f"<td>{s['median_a']:.2f}</td>"
            f"<td>{s['median_b']:.2f}</td>"
            f"<td>{s['adv_a']:+.2f}</td>"
            f"<td>{rel}</td>"
            f"<td>{s['dz']:+.3f}</td>"
            f"<td>{_fmt_p(s['t_p'])}</td>"
            f"<td>{_fmt_p(s['wil_p'])}</td>"
            + _winner_cell(s, name_a, name_b, alpha)
            + "</tr>"
        )
    head = (
        "<tr>"
        "<th>Task</th><th>Lang</th><th>Metric</th><th>n</th>"
        f"<th>{name_a}<br>(mean)</th><th>{name_b}<br>(mean)</th>"
        f"<th>{name_a}<br>(median)</th><th>{name_b}<br>(median)</th>"
        f"<th>&Delta; mean<br><span class='sub'>+&rArr;{name_a} better</span></th>"
        "<th>Rel.<br>impr.</th>"
        "<th>Cohen&rsquo;s d<sub>z</sub></th>"
        "<th>t-test<br>p</th><th>Wilcoxon<br>p</th><th>Winner</th>"
        "</tr>"
    )
    return f"<table class='cmp'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def build_dataset_table(dataset_rows, group_stats_fn, name_a, name_b, alpha):
    """Per-dataset detail table (grouped visually by task/language)."""
    rows = []
    for r in sorted(dataset_rows, key=lambda r: _group_sort_key((r["task"], r["language"], r["metric"]))):
        s = group_stats_fn(r["a"], r["b"], r["metric"])
        unit = " %" if r["metric"] in ZERO_TO_ONE_RANGE else ""
        rows.append(
            "<tr>"
            f"<td style='text-align:left'>{_task_display_name(r['task'] or '?')}</td>"
            f"<td>{(r['language'] or '?')}</td>"
            f"<td style='text-align:left'>{r['dataset']}</td>"
            f"<td>{r['metric']}{unit}</td>"
            f"<td>{s['n']}</td>"
            f"<td>{s['mean_a']:.2f}</td>"
            f"<td>{s['mean_b']:.2f}</td>"
            f"<td>{s['median_a']:.2f}</td>"
            f"<td>{s['median_b']:.2f}</td>"
            f"<td>{s['adv_a']:+.2f}</td>"
            f"<td>{_fmt_p(s['t_p'])}</td>"
            f"<td>{_fmt_p(s['wil_p'])}</td>"
            + _winner_cell(s, name_a, name_b, alpha)
            + "</tr>"
        )
    head = (
        "<tr>"
        "<th>Task</th><th>Lang</th><th>Dataset</th><th>Metric</th><th>n</th>"
        f"<th>{name_a}<br>(mean)</th><th>{name_b}<br>(mean)</th>"
        f"<th>{name_a}<br>(median)</th><th>{name_b}<br>(median)</th>"
        "<th>&Delta; mean</th>"
        "<th>t-test p</th><th>Wilcoxon p</th><th>Winner</th>"
        "</tr>"
    )
    return f"<table class='cmp'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def forest_plot(group_stats, name_a, name_b, alpha):
    """Forest plot of Cohen's dz (+95% CI). Positive => system A better."""
    keys = sorted(group_stats, key=_group_sort_key, reverse=True)  # reverse: top row first
    labels, dz, lo, hi, colors = [], [], [], [], []
    for key in keys:
        task, lang, metric = key
        s = group_stats[key]
        labels.append(f"{_task_display_name(task or '?')} · {lang or '?'} · {metric}")
        dz.append(s["dz"])
        lo.append(s["dz"] - s["dz_lo"])  # error bar lengths
        hi.append(s["dz_hi"] - s["dz"])
        if s["t_p"] is not None and not math.isnan(s["t_p"]) and s["t_p"] < alpha:
            colors.append(_FIRST_COLOR if s["winner"] == "A" else _SECOND_COLOR)
        else:
            colors.append(_NS_COLOR)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dz, y=labels, mode="markers",
        marker=dict(size=11, color=colors, line=dict(width=1, color="#333")),
        error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, thickness=1.5),
        hovertemplate="%{y}<br>d_z=%{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#333")
    fig.update_layout(
        title_text=(f"Standardized effect size (Cohen's d<sub>z</sub>, 95% CI) — "
                    f"right ⇒ {name_a} better, left ⇒ {name_b} better"),
        title_font_size=14,
        xaxis_title="Cohen's d_z  (paired effect size; |0.2| small, |0.5| medium, |0.8| large)",
        height=max(320, 46 * len(labels) + 140),
        width=950,
        template="plotly_white",
        margin=dict(l=320),
    )
    return fig


def grouped_bar_plots(group_stats, name_a, name_b):
    """One grouped-bar figure per metric (means + 95% CI), to avoid mixing scales."""
    by_metric = defaultdict(list)
    for key in group_stats:
        by_metric[key[2]].append(key)

    figs = []
    for metric in sorted(by_metric):
        keys = sorted(by_metric[metric], key=_group_sort_key)
        labels = [f"{_task_display_name(t or '?')} · {lang or '?'}" for (t, lang, _) in keys]
        mean_a = [group_stats[k]["mean_a"] for k in keys]
        mean_b = [group_stats[k]["mean_b"] for k in keys]
        # 95% CI half-widths (display-scaled), pre-computed in main()
        ci_a = [group_stats[k].get("ci_a", 0.0) for k in keys]
        ci_b = [group_stats[k].get("ci_b", 0.0) for k in keys]

        unit = " (%)" if metric in ZERO_TO_ONE_RANGE else ""
        lower_better = metric in LOWER_IS_BETTER
        direction = "lower is better" if lower_better else "higher is better"

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=name_a, x=labels, y=mean_a, marker_color=_FIRST_COLOR,
            error_y=dict(type="data", array=ci_a, thickness=1.2),
        ))
        fig.add_trace(go.Bar(
            name=name_b, x=labels, y=mean_b, marker_color=_SECOND_COLOR,
            error_y=dict(type="data", array=ci_b, thickness=1.2),
        ))
        fig.update_layout(
            title_text=f"{metric.upper()}{unit} — mean ± 95% CI ({direction})",
            title_font_size=14,
            barmode="group",
            xaxis_tickangle=30,
            yaxis_title=f"{metric.upper()}{unit}",
            height=460,
            width=max(640, 150 * len(labels) + 200),
            template="plotly_white",
        )
        figs.append((metric, fig))
    return figs


# ---------------------------------------------------------------------------
# Radar plots
# ---------------------------------------------------------------------------

def _primary_metric_by_task(keys):
    """Pick one representative metric per task (e.g. AST -> meteor)."""
    from collections import Counter, defaultdict as dd
    counts = dd(Counter)
    for task, _lang, metric in keys:
        counts[task][metric] += 1
    primary = {}
    for task, c in counts.items():
        override = _TASK_METRIC_OVERRIDE.get((task or "").upper())
        primary[task] = override if (override and override in c) else c.most_common(1)[0][0]
    return primary


def _radar_quality(mean_raw, metric):
    """Map a raw metric mean to a 0..100 'quality' score (higher = better).

    Every metric is normalised onto a common outward-is-better scale so that
    different tasks can share one radar without their native scales clashing.
    """
    if metric == "wer":
        q = 1.0 - mean_raw                 # WER (already clipped to <=1) -> accuracy
    elif metric == "flow_judge":
        q = (mean_raw + 1.0) / 6.0         # judge score in [-1, 5]
    elif metric in ("bleu", "temporal_regex"):
        q = mean_raw / 100.0               # already 0..100
    else:
        q = mean_raw                       # meteor, acc, label_match, format_check: 0..1
    return max(0.0, min(1.0, q)) * 100.0


def _radar_figure(axis_labels, qual_a, qual_b, name_a, name_b, title):
    """Build a two-trace radar (Scatterpolar) figure. Needs >= 3 axes."""
    # Close the polygon by repeating the first point.
    theta = axis_labels + [axis_labels[0]]
    ra = qual_a + [qual_a[0]]
    rb = qual_b + [qual_b[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=ra, theta=theta, name=name_a, fill="toself",
        line=dict(color=_FIRST_COLOR), opacity=0.7,
        hovertemplate="%{theta}<br>%{r:.1f}/100<extra>" + name_a + "</extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=rb, theta=theta, name=name_b, fill="toself",
        line=dict(color=_SECOND_COLOR), opacity=0.7,
        hovertemplate="%{theta}<br>%{r:.1f}/100<extra>" + name_b + "</extra>",
    ))
    fig.update_layout(
        title_text=title, title_font_size=14,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=480, width=640, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )
    return fig


def task_radar(group_samples, name_a, name_b):
    """Overall radar: one axis per task, outward = better (normalised quality).

    Returns a figure, or None when fewer than 3 tasks are available.
    """
    primary = _primary_metric_by_task(group_samples.keys())
    # Pool per-task across languages for the task's primary metric.
    task_a = defaultdict(list)
    task_b = defaultdict(list)
    for (task, _lang, metric), samples in group_samples.items():
        if metric != primary.get(task):
            continue
        task_a[task].extend(samples["a"])
        task_b[task].extend(samples["b"])

    tasks = sorted(task_a, key=lambda t: _task_display_name(t or ""))
    if len(tasks) < 3:
        return None

    labels = [f"{_task_display_name(t or '?')}<br>({primary[t]})" for t in tasks]
    qual_a = [_radar_quality(float(np.mean(task_a[t])), primary[t]) for t in tasks]
    qual_b = [_radar_quality(float(np.mean(task_b[t])), primary[t]) for t in tasks]
    return _radar_figure(
        labels, qual_a, qual_b, name_a, name_b,
        "Task profile — normalised quality (0–100, outward = better)",
    )


def dataset_radars_by_task(dataset_rows, name_a, name_b):
    """Per-task radars: one axis per {dataset, language} pair (>= 3 pairs).

    Returns a list of (task_display_name, figure).
    """
    primary = _primary_metric_by_task(
        (r["task"], r["language"], r["metric"]) for r in dataset_rows
    )
    by_task = defaultdict(list)
    for r in dataset_rows:
        if r["metric"] == primary.get(r["task"]):
            by_task[r["task"]].append(r)

    figs = []
    for task in sorted(by_task, key=lambda t: _task_display_name(t or "")):
        rows = sorted(by_task[task], key=lambda r: (_lang_sort_key(r["language"] or ""), r["dataset"]))
        if len(rows) < 3:
            continue
        metric = primary[task]
        labels = [f"{r['dataset']}<br>({r['language'] or '?'})" for r in rows]
        qual_a = [_radar_quality(float(np.mean(r["a"])), metric) for r in rows]
        qual_b = [_radar_quality(float(np.mean(r["b"])), metric) for r in rows]
        fig = _radar_figure(
            labels, qual_a, qual_b, name_a, name_b,
            f"{_task_display_name(task or '?')} ({metric}) — per dataset×language "
            f"(normalised, outward = better)",
        )
        figs.append((_task_display_name(task or "?"), fig))
    return figs


_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>System comparison: {name_a} vs {name_b}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #1a202c; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 17px; margin-top: 32px; border-bottom: 2px solid #e2e8f0; padding-bottom: 4px; }}
  .meta {{ font-size: 13px; color: #475569; margin: 6px 0 18px; }}
  .meta b {{ color: #1a202c; }}
  table.cmp {{ border-collapse: collapse; font-size: 12px; margin: 10px 0; }}
  table.cmp th, table.cmp td {{ border: 1px solid #e2e8f0; padding: 6px 9px; text-align: center; white-space: nowrap; }}
  table.cmp thead th {{ background: #3a5a8c; color: white; font-weight: 600; }}
  table.cmp .sub {{ font-weight: 400; font-size: 9px; opacity: .8; }}
  .warn {{ background: #fff3cd; border: 1px solid #ffe69c; border-radius: 6px;
           padding: 10px 14px; font-size: 12px; margin: 12px 0; }}
  .warn ul {{ margin: 6px 0 0; padding-left: 20px; }}
  .legend {{ font-size: 12px; margin: 8px 0; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 5px; margin-right: 16px; }}
  .legend i {{ width: 13px; height: 13px; border-radius: 3px; display: inline-block; }}
  .fig {{ margin: 14px 0 28px; }}
  details {{ margin: 8px 0; }}
  summary {{ cursor: pointer; font-weight: 600; }}
</style>
</head><body>
<h1>Pairwise comparison &mdash; <span style="color:{c1}">{name_a}</span> vs <span style="color:{c2}">{name_b}</span></h1>
<div class="meta">
  <b>A</b> = {name_a} &nbsp;|&nbsp; <b>B</b> = {name_b}<br>
  Paired tests on per-sample scores, pooled per <b>{{task, language, metric}}</b>.
  Significance threshold &alpha; = <b>{alpha}</b>. Stars: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05.
</div>
<div class="legend">
  <span><i style="background:{c1}"></i> {name_a} significantly better</span>
  <span><i style="background:{c2}"></i> {name_b} significantly better</span>
  <span><i style="background:{cns}"></i> not significant (p &ge; &alpha;)</span>
</div>
{warnings}
<h2>Summary per {{task, language}}</h2>
{summary_table}
<h2>Task profile (radar)</h2>
{task_radar}
<h2>Effect size (forest plot)</h2>
<div class="fig">{forest}</div>
<h2>Mean scores per metric</h2>
{bars}
<h2>Per-dataset breakdown</h2>
<details><summary>Show per-dataset table ({n_datasets} datasets)</summary>
{dataset_table}
</details>
<details><summary>Show per-task radars (one axis per dataset×language)</summary>
{dataset_radars}
</details>
</body></html>
"""


def _fig_div(fig):
    return f"<div class='fig'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"


def build_report(group_stats, group_samples, dataset_rows, warnings,
                 name_a, name_b, alpha, output_path):
    if warnings:
        warn_html = ("<div class='warn'><b>⚠ Warnings (data not perfectly aligned):</b><ul>"
                     + "".join(f"<li>{w}</li>" for w in warnings) + "</ul></div>")
    else:
        warn_html = "<div class='warn' style='background:#e6f4ea;border-color:#b7e1c4'>✓ All shared score files line up (same files, counts, and reference order).</div>"

    summary_html = build_summary_table(group_stats, name_a, name_b, alpha)

    forest_html = forest_plot(group_stats, name_a, name_b, alpha).to_html(
        full_html=False, include_plotlyjs=False)

    bars_html = "\n".join(_fig_div(fig) for _metric, fig in grouped_bar_plots(group_stats, name_a, name_b))

    # Overall task radar (>= 3 tasks)
    radar_fig = task_radar(group_samples, name_a, name_b)
    task_radar_html = _fig_div(radar_fig) if radar_fig else (
        "<p style='font-size:12px;color:#64748b'>Need at least 3 tasks for a radar plot.</p>")

    # Per-task dataset radars (>= 3 dataset×language pairs each)
    ds_radars = dataset_radars_by_task(dataset_rows, name_a, name_b)
    if ds_radars:
        dataset_radars_html = "\n".join(
            f"<h3 style='font-size:14px;margin:14px 0 0'>{task}</h3>{_fig_div(fig)}"
            for task, fig in ds_radars)
    else:
        dataset_radars_html = ("<p style='font-size:12px;color:#64748b'>No task has "
                               "at least 3 dataset×language pairs.</p>")

    def _stats_fn(a, b, metric):
        return paired_stats(a, b, metric, alpha)

    dataset_html = build_dataset_table(dataset_rows, _stats_fn, name_a, name_b, alpha)

    html = _HTML.format(
        name_a=name_a, name_b=name_b, alpha=alpha,
        c1=_FIRST_COLOR, c2=_SECOND_COLOR, cns=_NS_COLOR,
        warnings=warn_html, summary_table=summary_html, forest=forest_html,
        bars=bars_html, task_radar=task_radar_html, dataset_radars=dataset_radars_html,
        dataset_table=dataset_html, n_datasets=len(dataset_rows),
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Saved report: {output_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_console_summary(group_stats, name_a, name_b):
    print(f"\nPairwise comparison: A={name_a}  vs  B={name_b}\n")
    header = f"{'Task':<26}{'Lang':<8}{'Metric':<16}{'n':>6}  {'meanA':>8}{'meanB':>8}{'dz':>8}  {'t-p':>9}{'wil-p':>9}  Winner"
    print(header)
    print("-" * len(header))
    for key in sorted(group_stats, key=_group_sort_key):
        task, lang, metric = key
        s = group_stats[key]
        winner = {"A": name_a, "B": name_b, "tie": "tie"}[s["winner"]]
        print(f"{_task_display_name(task or '?'):<26}{(lang or '?'):<8}{metric:<16}{s['n']:>6}  "
              f"{s['mean_a']:>8.2f}{s['mean_b']:>8.2f}{s['dz']:>8.3f}  "
              f"{_fmt_p(s['t_p']):>9}{_fmt_p(s['wil_p']):>9}  {winner} {s['sig']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pairwise statistical comparison of two AudioBench systems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results_folder", help="Path to results folder (e.g. results/)")
    parser.add_argument("system_a", help="First system (sub-folder name or path)")
    parser.add_argument("system_b", help="Second system (sub-folder name or path)")
    parser.add_argument("--output_folder", default="plots/", help="Where to save the HTML report")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument(
        "--dont-check-ref", "--dont_check_ref", dest="dont_check_ref",
        action="store_true",
        help="Skip the strong check that reference texts line up in the same order",
    )
    parser.add_argument(
        "--no-clip-wer", "--no_clip_wer", dest="clip_wer", action="store_false",
        help="Do not clip per-sample WER above 100%% to 100%% "
             "(by default it is clipped so a few catastrophic samples do not "
             "dominate means/tests)",
    )
    args = parser.parse_args()

    name_a = Path(args.system_a).name
    name_b = Path(args.system_b).name

    group_samples, dataset_rows, warnings = compare(
        args.results_folder, args.system_a, args.system_b,
        check_ref=not args.dont_check_ref, clip_wer=args.clip_wer,
    )

    for w in warnings:
        print(f"WARNING: {w}")

    if not group_samples:
        sys.exit("No shared datasets with per-sample scores were found for the two systems.")

    # Compute pooled stats per group, plus CI of each mean for the bar plots.
    group_stats = {}
    for key, samples in group_samples.items():
        metric = key[2]
        s = paired_stats(samples["a"], samples["b"], metric, args.alpha)
        sf = 100.0 if metric in ZERO_TO_ONE_RANGE else 1.0
        for who, label in (("a", "ci_a"), ("b", "ci_b")):
            arr = np.asarray(samples[who], dtype=float)
            s[label] = (1.96 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr)) * sf
                        if len(arr) >= 2 else 0.0)
        group_stats[key] = s

    print_console_summary(group_stats, name_a, name_b)

    out = os.path.join(args.output_folder, f"comparison_{name_a}_vs_{name_b}.html")
    build_report(group_stats, group_samples, dataset_rows, warnings,
                 name_a, name_b, args.alpha, out)


if __name__ == "__main__":
    main()
