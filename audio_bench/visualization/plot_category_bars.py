"""Per-super-category grouped-bar PNGs + an overview-table PNG from a results/ folder.

Companion to ``plot_radar.py`` / ``plot_results_to_html.py``, used by the
``evaluate_on_dgx`` Airflow DAG's ``generate_html`` step. It writes, into
``--output_folder``:

* ``<SuperCat>.png`` for every super-category present (ASR, AST, QA, Others,
  Music, Sound), 2 subplots per row. Each subplot is one **language** (or, for
  ``Others``, one **sub-task**); within it the datasets sit on the x-axis with a
  grouped bar per model, showing the **original metric score** (WER, BLEU,
  flow_judge, acc as reported; 0-1 metrics ×100). The best model per dataset
  group is annotated and marked with a dotted horizontal line (best = lowest for
  WER, highest otherwise). ASR is capped at y=50 for readability.
* ``overview_table.png``: models (rows, sorted by avg rank) × the aggregate
  measures (minmax, zscore, avg_rank) plus each super-category's normalized score
  (or, with ``--by dataset``, one column per individual dataset — for trimmed
  non-default suites whose datasets all collapse into a single super-category).

Usage:
    python -m audio_bench.visualization.plot_category_bars results/ \
        --output_folder plots/ --show-all
"""
from __future__ import annotations

import argparse
import math
import os
from collections import OrderedDict, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from audio_bench.visualization.plot_results_to_html import (  # noqa: E402
    AGGREGATE_MEASURES,
    LOWER_IS_BETTER,
    _compute_overview_ranks,
    _display_score,
    _excluded_from_task_avg,
    _model_color_map,
    _super_category,
    _SUPER_CATEGORY_ORDER,
    load_all_scores,
)
from audio_bench.visualization.plot_radar import (  # noqa: E402
    _lang_suffix,
    _oriented_raw,
    _select_task_metrics,
)

_DPI = 150
_NCOLS = 2                       # 2 subplots per row
_ASR_YMAX = 50                   # cap ASR (WER) y-axis for readability
_GRID = "#e9e9e9"
_BEST_LINE = "#444444"
_DOTTED = (0, (1, 1.4))


def _short(model: str) -> str:
    """Compact model label for legends/table rows (drop the org prefix)."""
    return model.split("/")[-1]


def _lang_of(entry) -> str:
    lang = entry.get("language")
    return lang if lang and lang != "UNKNOWN" else "n.a."


def _group_of(entry, sc: str) -> str:
    """Subplot bucket: sub-task for 'Others', language otherwise."""
    if sc == "Others":
        return (entry.get("task") or "Unknown").title()
    return _lang_of(entry)


def _orig_value(entry) -> float:
    """Original metric score as reported (0-1 metrics scaled ×100 for display)."""
    return _display_score(entry["score"], entry["metric_name"])


def _style_ax(ax) -> None:
    """A cleaner, more modern look: no box, soft horizontal grid only."""
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(length=0, labelsize=7, colors="#444444")
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_category_bars(entries, out_folder) -> list[str]:
    """One PNG per super-category; a subplot per language (or sub-task for Others)."""
    entries = _select_task_metrics(entries)  # one metric per task (no double count)
    models = sorted({e["model_name"] for e in entries})
    colors = _model_color_map(models)
    written = []

    for sc in _SUPER_CATEGORY_ORDER:
        sc_entries = [e for e in entries if _super_category(e.get("task") or "") == sc]
        if not sc_entries:
            continue

        groups = sorted({_group_of(e, sc) for e in sc_entries})
        by_group = {}
        for g in groups:
            labels = OrderedDict()          # (dataset, metric) -> label
            vals = defaultdict(dict)        # cell -> model -> original score
            ascending = {}                  # cell -> True if lower is better
            for e in sc_entries:
                if _group_of(e, sc) != g:
                    continue
                k = (e["dataset_name"], e["metric_name"])
                labels.setdefault(k, f"{e['dataset_name']}\n({e['metric_name']})")
                vals[k][e["model_name"]] = _orig_value(e)
                ascending[k] = e["metric_name"] in LOWER_IS_BETTER
            by_group[g] = (list(labels.keys()), labels, vals, ascending)

        cat_models = [m for m in models
                      if any(m in v for _, _, vals, _ in by_group.values()
                             for v in vals.values())]
        if not cat_models:
            continue
        n_m = len(cat_models)

        n_g = len(groups)
        ncols = min(_NCOLS, n_g)
        nrows = math.ceil(n_g / ncols)
        max_groups = max(len(keys) for keys, _, _, _ in by_group.values())
        sub_w = max(5.5, max_groups * n_m * 0.17 + 2.0)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(sub_w * ncols, 3.7 * nrows + 1.3),
                                 squeeze=False)
        fig.patch.set_facecolor("white")

        group_w = 0.8
        bar_w = group_w / n_m
        for idx, g in enumerate(groups):
            ax = axes[idx // ncols][idx % ncols]
            _style_ax(ax)
            keys, labels, vals, ascending = by_group[g]
            present = [m for m in cat_models if any(m in vals[k] for k in keys)]
            x = np.arange(len(keys))
            for m in present:
                offs = -group_w / 2 + (cat_models.index(m) + 0.5) * bar_w
                y = [vals[k].get(m, np.nan) for k in keys]
                ax.bar(x + offs, y, bar_w, color=colors[m],
                       edgecolor="white", linewidth=0.5)
            # Best model per dataset group: dotted line spanning the group + label.
            for i, k in enumerate(keys):
                cell = {m: vals[k][m] for m in present if m in vals[k]}
                if not cell:
                    continue
                best = (min if ascending[k] else max)(cell, key=cell.get)
                bv = cell[best]
                ax.hlines(bv, x[i] - group_w / 2, x[i] + group_w / 2,
                          colors=_BEST_LINE, linestyles=_DOTTED, linewidth=1,
                          zorder=6)
                bx = x[i] - group_w / 2 + (cat_models.index(best) + 0.5) * bar_w
                ax.annotate(f"{bv:.1f}", (bx, bv), textcoords="offset points",
                            xytext=(0, 3), ha="center", va="bottom",
                            fontsize=6.5, fontweight="bold", color="#222222")
            ax.set_xticks(x)
            ax.set_xticklabels([labels[k] for k in keys], fontsize=7)
            ax.set_title(g, fontsize=11, fontweight="bold", color="#333333",
                         loc="left", pad=6)
            if sc == "ASR":
                ax.set_ylim(0, _ASR_YMAX)
            else:
                ax.margins(y=0.18)
        for idx in range(n_g, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m in cat_models]
        fig.legend(handles, [_short(m) for m in cat_models], loc="lower center",
                   ncol=min(5, n_m), fontsize=7.5, frameon=False)
        group_word = "sub-task" if sc == "Others" else "language"
        note = ";  WER — lower is better (y capped at 50)" if sc == "ASR" else ""
        fig.suptitle(f"{sc} — original scores by {group_word}"
                     f"   ·   best = dotted line{note}",
                     fontsize=14, fontweight="bold", color="#222222")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))

        out = os.path.join(out_folder, f"{sc}.png")
        fig.savefig(out, dpi=_DPI, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(out)
        print(f"[ok] wrote {out}")
    return written


def _fmt(v, decimals: int) -> str:
    """Format a measure, turning ±inf sentinels (missing data) into a dash."""
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{decimals}f}"


def _super_cat_scores(entries):
    """Per-model mean global-normalized score within each super-category (for the
    overview table's per-category columns). Returns (sc -> model -> float, [scs])."""
    entries = _select_task_metrics(entries)
    acc = defaultdict(lambda: defaultdict(list))  # sc -> model -> [values]
    for e in entries:
        if _excluded_from_task_avg(e):
            continue  # e.g. Arabic ASR is not folded into the ASR average
        sc = _super_category(e.get("task") or "")
        acc[sc][e["model_name"]].append(min(max(_oriented_raw(e) / 100.0, 0.0), 1.0))
    scs = [sc for sc in _SUPER_CATEGORY_ORDER if sc in acc]
    out = {sc: {m: float(np.mean(v)) for m, v in acc[sc].items()} for sc in acc}
    return out, scs


def _dataset_scores(entries):
    """Per-model mean global-normalized score for each individual dataset (for the
    overview table's per-dataset columns, used by trimmed non-default suites where a
    single super-category collapses every dataset into one column). Unlike
    ``_super_cat_scores`` this does NOT drop ``_excluded_from_task_avg`` datasets —
    the trimmed suite (e.g. the Arabic ASR datasets) IS what we want to show.
    Returns (label -> model -> float, [labels])."""
    entries = _select_task_metrics(entries)
    acc = defaultdict(lambda: defaultdict(list))  # label -> model -> [values]
    for e in entries:
        label = e["dataset_name"] + _lang_suffix(e)
        acc[label][e["model_name"]].append(min(max(_oriented_raw(e) / 100.0, 0.0), 1.0))
    labels = sorted(acc)
    out = {lbl: {m: float(np.mean(v)) for m, v in acc[lbl].items()} for lbl in acc}
    return out, labels


def plot_overview_table(entries, out_path, by="super") -> str | None:
    """Render the overview aggregate table (models × measures + per-column score) to PNG.

    ``by="super"`` (default) uses one column per super-category; ``by="dataset"`` uses
    one column per individual dataset — the latter for non-default trimmed suites whose
    datasets all share a single super-category (which would otherwise be one column).
    """
    data = _compute_overview_ranks(entries)
    if not data:
        print("[warn] no aggregate data for overview table")
        return None

    models = data["sorted_models"]
    if by == "dataset":
        col_scores, cols = _dataset_scores(entries)
        title = "Overview — aggregate measures & per-dataset score (normalized)"
    else:
        col_scores, cols = _super_cat_scores(entries)
        title = "Overview — aggregate measures & per-super-category score (normalized)"
    meas_dec = {"minmax": 3, "zscore": 2, "avg_rank": 1}
    meas_src = {
        "minmax": data["model_minmax"],
        "zscore": data["model_zscore"],
        "avg_rank": data["model_avg_rank"],
    }
    # Dataset labels ("Multilingual_TEDx [FR]") are far longer than super-category
    # labels ("ASR"); wrap the "[LANG]" suffix onto a second line and give each
    # column more width so the headers don't collide.
    col_labels = list(AGGREGATE_MEASURES) + [c.replace(" [", "\n[") for c in cols]

    rows = []
    for m in models:
        row = [_fmt(meas_src[meas].get(m), meas_dec[meas]) for meas in AGGREGATE_MEASURES]
        row += [_fmt(col_scores.get(c, {}).get(m), 2) for c in cols]
        rows.append(row)

    n_rows, n_cols = len(models), len(col_labels)
    per_col = 1.6 if by == "dataset" else 1.15
    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 + n_cols * per_col),
                                    max(2.0, 0.45 * n_rows + 1.0)))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    tbl = ax.table(cellText=rows, rowLabels=[_short(m) for m in models],
                   colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0 or c == -1:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f2f2f2")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[ok] wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_folder", help="Path to results folder (e.g. results/)")
    ap.add_argument("--output_folder", default="plots/",
                    help="Where to write the PNGs (default: plots/)")
    ap.add_argument("--show-all", "--show_all", dest="show_all", action="store_true",
                    help="Bypass BOTH curated filters (datasets + models), "
                         "matching plot_results_to_html --show-all.")
    ap.add_argument("--show_all_models", "--show-all-models", dest="show_all_models",
                    action="store_true",
                    help="Bypass the model allowlist/ignore patterns (show all "
                         "models). Ignored datasets are still filtered out.")
    ap.add_argument("--show_all_datasets", "--show-all-datasets", dest="show_all_datasets",
                    action="store_true",
                    help="Bypass _IGNORED_DATASETS (show ablation datasets). "
                         "The model allowlist/ignore patterns still apply.")
    # overview_table vs per-category bar PNGs. The DAG picks one per config: the default
    # suite gets the overview leaderboard (--overview-only); a trimmed non-default suite
    # gets the raw-metric bars (--no-overview), which already emit one PNG per super-
    # category actually present (so an ASR-only Arabic suite yields just ASR.png).
    outputs = ap.add_mutually_exclusive_group()
    outputs.add_argument("--overview-only", dest="overview_only", action="store_true",
                    help="write only overview_table.png (skip the per-category bar PNGs)")
    outputs.add_argument("--no-overview", dest="no_overview", action="store_true",
                    help="write only the per-category bar PNGs (skip overview_table.png); "
                         "one PNG per super-category present in the data")
    ap.add_argument("--by", choices=["super", "dataset"], default="super",
                    help="overview-table columns: one per super-category (default) or "
                         "one per individual dataset (for trimmed non-default suites "
                         "whose datasets collapse into a single super-category column)")
    args = ap.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    entries = load_all_scores(
        args.input_folder,
        show_all_models=args.show_all or args.show_all_models,
        show_all_datasets=args.show_all or args.show_all_datasets,
    )
    if not entries:
        print(f"No score files found in {args.input_folder}")
        return
    if not args.overview_only:
        plot_category_bars(entries, args.output_folder)
    if not args.no_overview:
        plot_overview_table(entries, os.path.join(args.output_folder, "overview_table.png"),
                            by=args.by)


if __name__ == "__main__":
    main()
