#!/usr/bin/env python3
"""Plot AudioBench evaluation results as a single interactive HTML report.

Generates one HTML file with an Overview table (all tasks × models ranked)
and per-task Summary sections (language-column tables with expandable
per-dataset sub-columns, and optionally violin plots).

Output: {output_folder}/index.html

Usage examples:
    python -m audio_bench.visualization.build_leaderboard results/
    python -m audio_bench.visualization.build_leaderboard results/ --violin
    python -m audio_bench.visualization.build_leaderboard results/ --output_folder my_plots/
    python -m audio_bench.visualization.build_leaderboard leaderboard/results/ --output_folder leaderboard/
"""

import argparse
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import plotly.express.colors as pxcolors
import plotly.graph_objects as go
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# What the report shows (datasets, models, display names and sizes) is set in
# leaderboard/config.yaml (override with --config).
_LEADERBOARD_CONFIG = Path(__file__).resolve().parents[2] / "leaderboard" / "config.yaml"


def _load_leaderboard_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    consortium = cfg.get("consortium_name", "")

    def fill(text):
        return str(text).replace("{consortium}", consortium)

    def name_map(key):
        return {str(k): fill(v) for k, v in (cfg.get(key) or {}).items()}

    return {
        "consortium_name": consortium,
        "ignored_datasets": set(cfg.get("ignored_datasets") or []),
        "avg_excluded_task_langs": {(str(t).upper(), str(l).upper())
                                    for t, l in cfg.get("avg_excluded_task_langs") or []},
        "consortium_models": name_map("consortium_models"),
        "ignored_model_patterns": [re.compile(fill(p))
                                   for p in cfg.get("ignored_model_patterns") or []],
        "model_renames": name_map("model_renames"),
        "organization_renames": name_map("organization_renames"),
        "model_sizes": [(re.compile(fill(p)), str(size))
                        for p, size in cfg.get("model_sizes") or []],
        "hf_links": bool(cfg.get("hf_links")),
        "hf_unlinked_models": [re.compile(fill(p))
                               for p in cfg.get("hf_unlinked_models") or []],
    }


def use_leaderboard_config(path):
    """(Re)load the leaderboard config into the module-level settings below."""
    global _IGNORED_DATASETS, _AVG_EXCLUDED_TASK_LANGS, CONSORTIUM_NAME, \
        ONLY_SHOW_CONSORTIUM_MODELS, _IGNORED_MODEL_PATTERNS, _MODEL_NAME_CORRECTIONS, \
        _MODEL_SIZE_OVERRIDES, _ORGANIZATION_RENAMES, _HF_LINKS, _HF_UNLINKED_MODELS
    cfg = _load_leaderboard_config(path)
    _IGNORED_DATASETS = cfg["ignored_datasets"]
    _AVG_EXCLUDED_TASK_LANGS = cfg["avg_excluded_task_langs"]
    CONSORTIUM_NAME = cfg["consortium_name"]
    ONLY_SHOW_CONSORTIUM_MODELS = cfg["consortium_models"]
    _IGNORED_MODEL_PATTERNS = cfg["ignored_model_patterns"]
    _MODEL_NAME_CORRECTIONS = cfg["model_renames"] | ONLY_SHOW_CONSORTIUM_MODELS
    _MODEL_SIZE_OVERRIDES = cfg["model_sizes"]
    _ORGANIZATION_RENAMES = cfg["organization_renames"]
    _HF_LINKS = cfg["hf_links"]
    _HF_UNLINKED_MODELS = cfg["hf_unlinked_models"]


use_leaderboard_config(_LEADERBOARD_CONFIG)


def _excluded_from_task_avg(entry):
    """True if this entry's (task, language) must be dropped from task averages."""
    task = str(entry.get("task") or "").upper()
    lang = str(entry.get("language") or "").upper()
    return (task, lang) in _AVG_EXCLUDED_TASK_LANGS


def _is_model_ignored(model_name):
    if ONLY_SHOW_CONSORTIUM_MODELS and CONSORTIUM_NAME in model_name:
        return model_name not in ONLY_SHOW_CONSORTIUM_MODELS.values()
    return any(p.search(model_name) for p in _IGNORED_MODEL_PATTERNS)


# Maps the (possibly shortened) display model name -> the model id (the
# results folder name). Populated by load_all_scores and used to show the
# model id as a hover tooltip in the tables.
_MODEL_CHECKPOINT = {}

# Maps the display model name -> the model_name of its score files (its
# Hugging Face id when hf_links is set). Populated like _MODEL_CHECKPOINT.
_MODEL_HF_ID = {}


def _model_hf_url(m):
    """Hugging Face page of model *m*, or None (hf_links off, or unlinked model)."""
    if not _HF_LINKS or any(p.search(m) for p in _HF_UNLINKED_MODELS):
        return None
    hf_id = _MODEL_HF_ID.get(m, m)
    return f"https://huggingface.co/{hf_id}" if "/" in hf_id else None


def _model_name_td(m):
    """Render the model-name table cell, showing the model id on hover and
    linking to the model's Hugging Face page when it has one."""
    model_id = _MODEL_CHECKPOINT.get(m, m)
    title_attr = f' title="{html.escape(model_id, quote=True)}"' if model_id != m else ""
    url = _model_hf_url(m)
    name = (f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{m}</a>'
            if url else m)
    return f'<td class="mname"{title_attr}>{name}</td>'

LOWER_IS_BETTER = {"wer"}
ZERO_TO_ONE_RANGE = {"wer", "meteor", "acc"}

_TASK_METRIC_OVERRIDE = {
    "AST": "meteor",
}

# Task display name overrides (raw task field → display name).
# Tasks not listed here use str.title() automatically.
_TASK_DISPLAY = {
    "ASR": "ASR",
    "AST": "AST",
}


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


# Task display order for the overview table columns.
# Tasks not listed here are appended alphabetically after the listed ones.
_LANG_TABLE_TASK_ORDER = [
    "ASR",
    "AST",
    "QUESTION ANSWERING",
    "MUSIC QUESTION ANSWERING",
    "EMOTION RECOGNITION",
    "GENDER RECOGNITION",
    "AGE RECOGNITION",
]
_LANG_TABLE_TASK_INDEX = {t: i for i, t in enumerate(_LANG_TABLE_TASK_ORDER)}


def _lang_table_task_sort_key(task: str):
    idx = _LANG_TABLE_TASK_INDEX.get(task.upper(), len(_LANG_TABLE_TASK_ORDER))
    return (idx, task)


def _group_by_super_category(entries):
    """Return {super_cat: {task_display: [entries]}}, super-categories in display order."""
    tmp = defaultdict(lambda: defaultdict(list))
    for e in entries:
        raw = e.get("task", "")
        if raw:
            sc = _super_category(raw)
            task = _task_display_name(raw)
            tmp[sc][task].append(e)
    result = {}
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


MISSING_COLOR = "#e0e0e0"
RANK_COLORS = {
    "first": "#5dade2",        # sky blue
    "second": "#82e0aa",       # green
    "last": "#e74c3c",         # bold red
    "before_last": "#fdcb6e",  # amber
}

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_all_scores(input_folder, show_all_models=False, show_all_datasets=False):
    """Scan input_folder/{model_dir}/**/*_score.json and return list of entry dicts.

    Supports the new results/ directory structure where score files may be nested
    in language subdirectories (e.g. results/model/FR/fleurs_score.json) and each
    file contains multiple metrics listed in data["metrics"].

    The two curated-benchmark filters can be bypassed independently:

    * ``show_all_datasets`` True bypasses ``_IGNORED_DATASETS`` (ablation datasets
      are shown as-is).
    * ``show_all_models`` True bypasses the model allowlist/ignore patterns
      (all models are shown, not just the curated consortium ones).
    """
    entries = []
    input_path = Path(input_folder)

    for model_dir in sorted(input_path.iterdir()):
        if not model_dir.is_dir():
            continue
        # Skip any "exclude" subfolder (a place to stash results that should
        # not appear in the plots) at the top level of results/.
        if model_dir.name.lower() == "exclude":
            continue
        model_id = model_dir.name

        for filepath in sorted(model_dir.rglob("*_score.json")):
            # Also skip score files nested under an "exclude" component
            # (e.g. results/<model>/exclude/... ).
            if any(part.lower() == "exclude"
                   for part in filepath.relative_to(model_dir).parts):
                continue
            dataset_name = filepath.name.removesuffix("_score.json")
            if not show_all_datasets and dataset_name in _IGNORED_DATASETS:
                continue

            try:
                data = json.loads(filepath.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            metrics = data.get("metrics", [])
            if not metrics:
                continue

            raw_model_name = data.get("model_name", model_id)
            model_name = _MODEL_NAME_CORRECTIONS.get(raw_model_name, raw_model_name)
            if not show_all_models and _is_model_ignored(model_name):
                continue
            # Remember the model id (results folder name) for the hover tooltip.
            _MODEL_CHECKPOINT.setdefault(model_name, model_id)
            _MODEL_HF_ID.setdefault(model_name, raw_model_name)
            task = data.get("task")
            language = data.get("language")
            sub_task = data.get("sub_task")

            for metric_name in metrics:
                # {"score": float, "all_scores": [per-sample floats], "std": float}
                raw_score = data.get(metric_name)
                score = raw_score.get("score") if isinstance(raw_score, dict) else None
                if not isinstance(score, (int, float)):
                    continue
                all_scores = raw_score.get("all_scores")
                std = raw_score.get("std")
                n = len(all_scores) if all_scores else None

                entry = {
                    "model_id": model_id,
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

    return _finalize_model_names(entries)


_STEP_SUFFIX_RE = re.compile(r"_step_\d+(?:-last)?$")


def _rename_organization(name):
    org, sep, rest = name.partition("/")
    return _ORGANIZATION_RENAMES[org] + sep + rest if sep and org in _ORGANIZATION_RENAMES else name


def _finalize_model_names(entries):
    """Display-name passes run after the filters (so they can't hide a model):

    The results folder name stays in the hover tooltip (_MODEL_CHECKPOINT)."""
    names = {e["model_name"] for e in entries}
    stems = Counter(_STEP_SUFFIX_RE.sub("", _rename_organization(n)) for n in names)
    renames = {}
    for n in names:
        new = _rename_organization(n)
        stem = _STEP_SUFFIX_RE.sub("", new)
        if stems[stem] == 1:
            new = stem
        if new != n:
            renames[n] = new
    for e in entries:
        e["model_name"] = renames.get(e["model_name"], e["model_name"])
    for old, new in renames.items():
        for names_map in (_MODEL_CHECKPOINT, _MODEL_HF_ID):
            if old in names_map:
                names_map[new] = names_map.pop(old)
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
                    if lang_key is None:
                        # Plain aggregate (task mean): drop languages excluded from
                        # a task's average, e.g. Arabic ASR. They still surface via
                        # the by_language path as their own sub-column.
                        matching = [e for e in matching if not _excluded_from_task_avg(e)]
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


# ---------------------------------------------------------------------------
# Symbolic scores (client-side dataset filtering)
# ---------------------------------------------------------------------------
#
# The tables are built from *symbolic* scores: each leaf score (one model on
# one dataset) is a ``SymScore`` and every mean computed from it by the table
# code (``sum(scores) / len(scores)``) or display scaling (``_display_score``)
# becomes a node of an expression graph.  Each rendered cell records the node
# it shows, so the report JS can re-evaluate every cell when datasets are
# unchecked in the sidebar.  Comparisons and formatting use the concrete value,
# so the Python-side layout, sorting and initial rendering are unchanged.

_SYM_NODES = []        # node id -> JSON spec (see SymScore)
_SYM_MEMO = {}         # dedup key -> SymScore
_SYM_DATASETS = []     # dataset index -> dataset key (task, language, dataset_name)
_SYM_DATASET_IDX = {}

# Token delimiters embedded in rendered strings; _resolve_sym_tokens() turns
# them into data-* attributes. Never valid in HTML, so they cannot clash.
_TOK_START, _TOK_SEP, _TOK_END = "\x00", "\x01", "\x02"


class SymScore:
    """A score value that remembers how it was computed.

    Node specs: ``["L", dataset_idx, value, asc, [n, sum, sumsq, std]]`` for a
    leaf, ``["M", [child ids]]`` for a mean and ``["D", child_id, pct]`` for a
    display score (x100 when *pct*, clamped to 100).
    """
    __slots__ = ("id", "val", "is_leaf")

    def __init__(self, spec, val, memo_key=None):
        self.id = len(_SYM_NODES)
        self.val = val
        self.is_leaf = spec[0] == "L"
        _SYM_NODES.append(spec)
        if memo_key is not None:
            _SYM_MEMO[memo_key] = self

    @classmethod
    def leaf(cls, entry):
        key = (_task_display_name(entry.get("task") or ""),
               (entry.get("language") or "UNKNOWN").upper(),
               entry["dataset_name"])
        if key not in _SYM_DATASET_IDX:
            _SYM_DATASET_IDX[key] = len(_SYM_DATASETS)
            _SYM_DATASETS.append(key)
        pooled = entry.get("all_scores") or []
        stats = [len(pooled), float(sum(pooled)), float(sum(x * x for x in pooled)),
                 entry.get("std")]
        spec = ["L", _SYM_DATASET_IDX[key], entry["score"],
                entry["metric_name"] in LOWER_IS_BETTER, stats]
        return cls(spec, entry["score"])

    @classmethod
    def mean(cls, items):
        key = ("M",) + tuple(i.id for i in items)
        if key in _SYM_MEMO:
            return _SYM_MEMO[key]
        return cls(["M", [i.id for i in items]],
                   sum(i.val for i in items) / len(items), key)

    @classmethod
    def display(cls, child, pct):
        key = ("D", child.id, pct)
        if key in _SYM_MEMO:
            return _SYM_MEMO[key]
        return cls(["D", child.id, pct],
                   min(child.val * 100 if pct else child.val, 100), key)

    # sum(scores) -> _SymSum, then / len(scores) -> mean node
    def __radd__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return _SymSum([self])
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, SymScore):
            return _SymSum([self, other])
        return NotImplemented

    def __float__(self):
        return float(self.val)

    def __lt__(self, other): return self.val < _sym_val(other)
    def __le__(self, other): return self.val <= _sym_val(other)
    def __gt__(self, other): return self.val > _sym_val(other)
    def __ge__(self, other): return self.val >= _sym_val(other)

    def __format__(self, spec):
        return sym_token("V", self.id, format(self.val, spec))


class _SymSum:
    """Intermediate of ``sum(sym_scores)``; only valid divided by its length."""
    __slots__ = ("items",)

    def __init__(self, items):
        self.items = items

    def __add__(self, other):
        if isinstance(other, SymScore):
            return _SymSum(self.items + [other])
        return NotImplemented

    def __truediv__(self, k):
        if k != len(self.items):
            raise ValueError("symbolic sums may only be averaged over their own items")
        return SymScore.mean(self.items)


def _sym_val(x):
    return x.val if isinstance(x, SymScore) else x


def sym_token(kind, node_id, text, *args):
    """Embed a token for node *node_id* displaying *text* (see _resolve_sym_tokens)."""
    head = ":".join([kind, str(node_id)] + [str(a) for a in args])
    return f"{_TOK_START}{head}{_TOK_SEP}{text}{_TOK_END}"


_SYM_TOKEN_RE = re.compile(
    re.escape(_TOK_START) + r"([^\x01]*)" + re.escape(_TOK_SEP) + r"([^\x02]*)" + re.escape(_TOK_END),
    re.S)
_TD_RE = re.compile(r"<td([^>]*)>(.*?)</td>", re.S)
_TITLE_RE = re.compile(r' title="([^"]*)"')


def _resolve_sym_tokens(raw_html):
    """Replace symbolic tokens in rendered table cells by data-* attributes.

    * ``data-f``  -- node id of the value shown in the cell
    * ``data-ci`` -- present when the cell shows a CI (value: 1 for x100 metrics)
    * ``data-tt`` -- tooltip template (from the title), tokens written as
      ``[[kind:id:args]]``; the report JS fills it in
    Tokens in the cell content are replaced by their initial text.
    """
    def plain(text):
        return _SYM_TOKEN_RE.sub(lambda m: m.group(2), text)

    def td(match):
        attrs, content = match.group(1), match.group(2)
        extra = ""
        first_v = next((t.group(1) for t in _SYM_TOKEN_RE.finditer(content)
                        if t.group(1).startswith("V:")), None)
        if first_v:
            extra += f' data-f="{first_v.split(":")[1]}"'
        ci = next((t.group(1) for t in _SYM_TOKEN_RE.finditer(content)
                   if t.group(1).startswith("CI:")), None)
        if ci:
            extra += f' data-ci="{ci.split(":")[2]}"'

        def title(tm):
            text = tm.group(1)
            if _TOK_START not in text:
                return tm.group(0)
            template = _SYM_TOKEN_RE.sub(lambda t: f"[[{t.group(1)}]]", text)
            return f' data-tt="{html.escape(template, quote=True)}"'
        attrs = _TITLE_RE.sub(title, attrs)
        return f"<td{attrs}{extra}>{plain(content)}</td>"

    out = _TD_RE.sub(td, raw_html)
    if _TOK_START in out:
        raise ValueError("symbolic score rendered outside a table cell")
    return out


def _symbolize_entries(entries):
    """Return copies of *entries* whose score is a leaf ``SymScore``."""
    return [dict(e, score=SymScore.leaf(e)) for e in entries]


def _display_score(score, metric):
    """Format score for display: multiply by 100 for 0-1 range metrics, clamp to 100."""
    if isinstance(score, SymScore):
        return SymScore.display(score, metric in ZERO_TO_ONE_RANGE)
    disp = score * 100 if metric in ZERO_TO_ONE_RANGE else score
    return min(disp, 100)


def _metric_label(metric):
    """Column label of a metric, e.g. 'WER %'."""
    return metric.upper() + (" %" if metric in ZERO_TO_ONE_RANGE else "")


def _score_cell(score, metric, model, sub_lines=()):
    """(html, tooltip) of the table cell showing a symbolic *score*.

    The report JS fills in the value, its CI and rank colour, and the tooltip:
    model, rank, value and CI, then *sub_lines* (see ``_tooltip_subline``).
    """
    disp = _display_score(score, metric)
    pct = int(metric in ZERO_TO_ONE_RANGE)
    tip = (f"{model}\n" + sym_token("R", disp.id, "", "p") + f"{disp:.2f}"
           + sym_token("C", disp.id, "", pct, "m"))
    return f"{disp:.2f}" + sym_token("CI", disp.id, "", pct), "\n".join([tip, *sub_lines])


def _tooltip_subline(name, score, metric):
    """One sub-item tooltip line, e.g. 'FLEURS: 18.50±1.20 (3e)' once filled in by the JS.

    Used to enumerate the datasets/languages hidden behind an expandable cell.
    """
    disp = _display_score(score, metric)
    pct = int(metric in ZERO_TO_ONE_RANGE)
    return (f"{name}: {disp:.2f}" + sym_token("C", disp.id, "", pct, "s")
            + sym_token("R", disp.id, "", "s"))


def _breakdown(model, parts):
    """Tooltip lines giving *model*'s score in each ``(label, scores, metric)`` part."""
    return [_tooltip_subline(label, scores[model], metric)
            for label, scores, metric in parts if model in scores]


def _sub_columns(parts):
    """The ``(header, cells)`` sub-columns of the ``(label, scores, metric)`` parts."""
    return [(label, {m: _score_cell(v, metric, m) for m, v in scores.items()})
            for label, scores, metric in parts]


def _sort_ascending(metric):
    """Return True if lower is better for this metric."""
    return metric in LOWER_IS_BETTER


def _two_line_label(label):
    """Insert <br> before a trailing ' (...)' suffix for two-line table headers."""
    if label.endswith(')') and ' (' in label:
        idx = label.rfind(' (')
        return label[:idx] + '<br>' + label[idx+1:]
    return label


def _extract_model_size(model_name):
    """Extract size like '7B' from model name by matching patterns like '_7b' or '_3b_'."""
    for pattern, size in _MODEL_SIZE_OVERRIDES:
        if pattern.search(model_name):
            return size
    match = re.search(r'(\d+)[bB](?:_|$)', model_name)
    return f"{match.group(1)}B" if match else ""


def _most_common_metric(entries):
    """Return the most frequent metric_name among *entries*."""
    counts = defaultdict(int)
    for e in entries:
        counts[e["metric_name"]] += 1
    return max(counts, key=counts.get)


def _task_metric(task, entries):
    """The metric a task is ranked on: its override (``_TASK_METRIC_OVERRIDE``)
    when some entry has it, else the most common metric of *entries*."""
    override = _TASK_METRIC_OVERRIDE.get(task.upper())
    if override and any(e["metric_name"] == override for e in entries):
        return override
    return _most_common_metric(entries)


def _mean_score(entries):
    """Mean score of *entries* (a mean node when the scores are symbolic)."""
    return sum(e["score"] for e in entries) / len(entries)


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
        hib = {m: (100.0 - float(s)) if asc else float(s) for m, s in scores.items()}
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


def _agg_payload_value(v):
    """A symbolic score is referenced by node id ({"n": id}), a float inlined."""
    return {"n": v.id} if isinstance(v, SymScore) else float(v)


def _agg_payload_html(tbl_id, table_aggregates, item_model_score, item_ascending,
                      item_model_rank_score=None):
    """Embed the per-item display scores behind a table's aggregate columns.

    The experiment filter reads this JSON to recompute Avg Rank / Min-Max /
    Z-Score (and re-sort rows) over the currently visible models only.
    *item_model_rank_score* optionally gives the (unclamped) scores the Avg
    Rank is computed on, when they differ from the display scores.
    """
    payload = {
        "aggs": [
            {"name": a,
             "digits": int(_AGG_META[a]["fmt"].strip(".f")),
             "hib": _AGG_META[a]["higher_is_better"]}
            for a in table_aggregates
        ],
        "items": [],
    }
    for item, scores in item_model_score.items():
        entry = {"asc": bool(item_ascending[item]),
                 "s": {m: _agg_payload_value(v) for m, v in scores.items()}}
        if item_model_rank_score is not None:
            entry["r"] = {m: _agg_payload_value(v)
                          for m, v in item_model_rank_score[item].items()}
        payload["items"].append(entry)
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (f'<script type="application/json" class="agg-data" '
            f'data-table="{tbl_id}">{blob}</script>')


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

    Returns None without data, else a dict with keys:

    * ``task_metric`` – task -> chosen metric name
    * ``all_models`` – sorted model list
    * ``task_model_score`` – task -> model -> score
    * ``column_tasks`` – overview columns: tasks, or super-categories grouping 2+ tasks
    * ``group_members`` – super-category column -> its member tasks
    * ``task_ascending`` – column -> True when lower is better
    * ``task_disp_score`` – column -> model -> display score
    * ``task_subcols`` / ``task_subcol_scores`` / ``task_subcol_metric`` – column ->
      its sub-columns, their scores (model -> score) and metric
    * ``model_avg_rank`` – model -> avg rank (float, or ``float('inf')``)
    * ``model_minmax`` – model -> min-max normalized score (float, or ``float('-inf')``)
    * ``model_zscore`` – model -> z-score normalized score (float, or ``float('-inf')``)
    * ``sorted_models`` – models sorted by the *sort_by* aggregate
    """
    agg = aggregate_entries(entries, by_language=False)
    if not agg:
        return None

    # Group by task
    task_entries = defaultdict(list)
    for e in agg:
        task_entries[e["task"]].append(e)

    # For each task pick the most common metric
    task_metric = {task: _task_metric(task, ents) for task, ents in task_entries.items()}

    tasks = sorted(task_metric.keys())
    if not tasks:
        return None

    all_models = sorted({e["model_name"] for e in agg})

    # Build score lookup: task -> model -> score
    task_model_score = {}
    for task in tasks:
        metric = task_metric[task]
        model_scores = {}
        for m in all_models:
            matching = [
                e for e in task_entries[task]
                if e["model_name"] == m and e["metric_name"] == metric
            ]
            if matching:
                model_scores[m] = _mean_score(matching)
        task_model_score[task] = model_scores

    # --- Task-based column layout (one column per task) ---
    ordered_tasks = sorted(tasks, key=_lang_table_task_sort_key)

    if allowed_super_cats is not None:
        allowed = {s.upper() for s in allowed_super_cats}
        ordered_tasks = [t for t in ordered_tasks if _super_category(t) in allowed]

    # Group tasks by super-category. Super-cats with 2+ tasks get a merged
    # pseudo-task column (like the existing "Others" group); single-task
    # super-cats keep their task column directly.
    sc_to_tasks = defaultdict(list)
    for t in ordered_tasks:
        sc_to_tasks[_super_category(t)].append(t)

    ordered_scs = [sc for sc in _SUPER_CATEGORY_ORDER if sc in sc_to_tasks]
    for sc in sc_to_tasks:
        if sc not in ordered_scs:
            ordered_scs.append(sc)

    column_tasks = []
    group_members = {}  # pseudo-task label -> list of member tasks
    for sc in ordered_scs:
        members = sc_to_tasks[sc]
        if not members:
            continue
        if sc == "Others":
            column_tasks.append("Others")
            group_members["Others"] = members
        elif len(members) == 1:
            column_tasks.append(members[0])
        else:
            column_tasks.append(sc)
            group_members[sc] = members

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
        task_lang_scores[task][lang][m] = e["score"]
        task_languages[task].add(lang)
    task_languages = {t: sorted(langs, key=_lang_sort_key) for t, langs in task_languages.items()}

    # --- Per-task sub-columns ---
    # ASR: sub-columns = languages (avg across datasets per language)
    # Others: sub-columns = lang-prefixed datasets
    task_subcols = {}       # task -> list of sub-column display names
    task_subcol_scores = {} # task -> subcol -> model -> score
    task_subcol_metric = {} # task -> subcol -> metric

    for task in ordered_tasks:
        metric = task_metric[task]
        sc = _super_category(task)

        if sc in {"ASR", "QA"}:
            subcols = task_languages.get(task, [])
            subcol_scores = {lang: task_lang_scores[task].get(lang, {}) for lang in subcols}
            subcol_metrics = {lang: metric for lang in subcols}
        elif sc in {"Others", "Music", "Sound"}:
            grouped = defaultdict(list)
            for e in entries:
                if (_task_display_name(e.get("task", "")) == task
                        and e["metric_name"] == metric):
                    key = e.get("sub_task") or (e.get("language") or "UNKNOWN").upper()
                    grouped[(key, e["model_name"])].append(e)
            subcols = sorted({k[0] for k in grouped})
            subcol_scores = defaultdict(dict)
            subcol_metrics = {}
            for st in subcols:
                subcol_metrics[st] = metric
                for m in all_models:
                    ents = grouped.get((st, m))
                    if ents:
                        subcol_scores[st][m] = _mean_score(ents)
            subcol_scores = dict(subcol_scores)
        else:
            # AST and any future SC: per-dataset (one dataset per language for AST).
            grouped = defaultdict(list)
            for e in entries:
                if (_task_display_name(e.get("task", "")) == task
                        and e["metric_name"] == metric):
                    lang = (e.get("language") or "UNKNOWN").upper()
                    grouped[(e["dataset_name"], lang, e["model_name"])].append(e)
            ds_lang_pairs = sorted({(k[0], k[1]) for k in grouped})
            multi_lang = len({p[1] for p in ds_lang_pairs}) >= 2
            subcols = []
            subcol_scores = defaultdict(dict)
            subcol_metrics = {}
            for ds, lang in ds_lang_pairs:
                display = f"{lang} \u00b7 {ds}" if multi_lang else ds
                subcols.append(display)
                subcol_metrics[display] = metric
                for m in all_models:
                    ds_entries = grouped.get((ds, lang, m))
                    if ds_entries:
                        subcol_scores[display][m] = _mean_score(ds_entries)
            subcol_scores = dict(subcol_scores)

        task_subcols[task] = subcols
        task_subcol_scores[task] = subcol_scores
        task_subcol_metric[task] = subcol_metrics

    # Per-task display scores (for ranking and aggregates)
    task_disp_score = {}  # task -> model -> display_score (float)
    task_ascending = {}
    for task in ordered_tasks:
        metric = task_metric[task]
        task_ascending[task] = _sort_ascending(metric)
        task_disp_score[task] = {}
        for m, val in task_model_score.get(task, {}).items():
            task_disp_score[task][m] = _display_score(val, metric)

    # --- Group pseudo-tasks: each aggregates 2+ member tasks of a super-category ---
    # The main cell is the mean of member-task display scores (mixed metrics, so
    # treat as a rough aggregate).
    # For "Others", sub-columns are the member tasks themselves (mixed task kinds).
    # For other super-cat groups (QA, Music, Sound), sub-columns are the individual
    # datasets from each member task (dataset-level drill-down, like a regular task).
    for group_label, members in group_members.items():
        disp = {}
        for m in all_models:
            vals = [task_disp_score[t][m] for t in members if m in task_disp_score[t]]
            if vals:
                disp[m] = sum(vals) / len(vals)
        task_disp_score[group_label] = disp
        task_ascending[group_label] = all(task_ascending[t] for t in members)

        if group_label in {"Others", "Music", "Sound"}:
            subcols = []
            subcol_scores = {}
            subcol_metric = {}
            for t in members:
                metric = task_metric[t]
                label = f"{t} ({_metric_label(metric)})"
                subcols.append(label)
                subcol_scores[label] = dict(task_model_score.get(t, {}))
                subcol_metric[label] = metric
        elif group_label == "QA":
            # Per-language sub-columns: average across member tasks.
            lang_to_model_vals = defaultdict(lambda: defaultdict(list))
            for t in members:
                for lang, model_map in task_lang_scores.get(t, {}).items():
                    for mdl, score in model_map.items():
                        lang_to_model_vals[lang][mdl].append(score)
            subcols = sorted(lang_to_model_vals.keys(), key=_lang_sort_key)
            subcol_scores = {}
            subcol_metric = {}
            metric_counts = defaultdict(int)
            for t in members:
                metric_counts[task_metric[t]] += 1
            group_metric = max(metric_counts, key=metric_counts.get) if metric_counts else None
            for lang in subcols:
                subcol_scores[lang] = {
                    mdl: sum(vals) / len(vals)
                    for mdl, vals in lang_to_model_vals[lang].items()
                }
                subcol_metric[lang] = group_metric
        else:
            # Fallback: concatenate each member task's sub-columns.
            subcols = []
            subcol_scores = {}
            subcol_metric = {}
            for t in members:
                for sc_label in task_subcols.get(t, []):
                    label = sc_label
                    if label in subcol_scores:
                        label = f"{sc_label} [{t}]"
                    subcols.append(label)
                    subcol_scores[label] = task_subcol_scores[t].get(sc_label, {})
                    subcol_metric[label] = task_subcol_metric[t].get(sc_label)

        task_subcols[group_label] = subcols
        task_subcol_scores[group_label] = subcol_scores
        task_subcol_metric[group_label] = subcol_metric

    # Per-task ranks (ranked columns = column_tasks, which includes "Others")
    task_model_rank = {}
    for task in column_tasks:
        asc = task_ascending.get(task, False)
        scores = task_disp_score.get(task, {})
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=not asc)
        task_model_rank[task] = {m: rank + 1 for rank, (m, _) in enumerate(ranked)}

    # Average rank per model (across column_tasks)
    model_avg_rank = {}
    for m in all_models:
        ranks = [task_model_rank[t][m] for t in column_tasks if m in task_model_rank.get(t, {})]
        model_avg_rank[m] = sum(ranks) / len(ranks) if ranks else float("inf")

    # --- Normalized aggregate scores (Min-Max and Z-Score) ---
    disp_for_norm = {t: task_disp_score.get(t, {}) for t in column_tasks}
    asc_for_norm = {t: task_ascending.get(t, False) for t in column_tasks}
    model_minmax, model_zscore = _compute_normalized_scores(
        all_models, disp_for_norm, asc_for_norm,
    )

    agg_values = {"avg_rank": model_avg_rank, "minmax": model_minmax, "zscore": model_zscore}
    sorted_models = _sort_models_by_aggregate(all_models, agg_values, sort_by)

    return {
        "task_metric": task_metric,
        "all_models": all_models,
        "task_model_score": task_model_score,
        "column_tasks": column_tasks,
        "group_members": group_members,
        "task_ascending": task_ascending,
        "task_disp_score": task_disp_score,
        "task_subcols": task_subcols,
        "task_subcol_scores": task_subcol_scores,
        "task_subcol_metric": task_subcol_metric,
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
                             overview_data=None, figure_aggregates=None,
                             allowed_super_cats=None):
    """Scatter plot(s) of aggregate score vs model size.

    Generates one figure per measure in *figure_aggregates*.
    If *overview_data* is provided, reuses it instead of recomputing.
    """
    if figure_aggregates is None:
        figure_aggregates = ["avg_rank"]

    data = overview_data or _compute_overview_ranks(
        entries, allowed_super_cats=allowed_super_cats)
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

def _render_table(tbl_id, models, aggregates, columns):
    """HTML lines of a score table, one row per model of *models*.

    *columns* lists ``(key, header, cells, subs)``: *cells* maps a model to the
    ``(html, tooltip)`` of its cell (see ``_score_cell``), *subs* is the
    breakdown hidden behind a [+] toggle, as ``(header, cells)`` sub-columns.
    Aggregate cells are left empty: the report JS computes them, like every
    value, CI, rank colour and tooltip, and sorts the rows.
    """
    def td(cell, attrs=""):
        if cell is None:
            return f'<td{attrs} style="background:{MISSING_COLOR}">-</td>'
        value, tip = cell
        title = f' title="{tip}"' if tip else ""
        return f"<td{attrs}{title}>{value}</td>"

    top = ['<th rowspan="2">Model</th>', '<th rowspan="2">Size</th>',
           f'<th colspan="{len(aggregates)}">Aggregation</th>']
    bottom = [f"<th>{_AGG_META[a]['label']}</th>" for a in aggregates]
    for key, header, _, subs in columns:
        if subs:
            grp = _slug(key)
            top.append(f'<th rowspan="2">{header} <button class="toggle-btn" '
                       f'onclick="toggleCols(this,\'{tbl_id}\',\'{grp}\')">+</button></th>')
            top += [f'<th rowspan="2" class="lang-col" data-group="{grp}">{h}</th>' for h, _ in subs]
        else:
            top.append(f'<th rowspan="2">{header}</th>')

    lines = [f'<table class="ov-tbl" id="{tbl_id}">',
             "<thead><tr>" + "".join(top) + "</tr><tr>" + "".join(bottom) + "</tr></thead>",
             "<tbody>"]
    for m in models:
        lines.append(f'<tr data-model="{html.escape(m, quote=True)}">')
        lines.append(_model_name_td(m))
        lines.append(f"<td>{_extract_model_size(m)}</td>")
        lines += [f'<td data-agg="{a}"></td>' for a in aggregates]
        for key, _, cells, subs in columns:
            lines.append(td(cells.get(m)))
            attrs = f' class="lang-col" data-group="{_slug(key)}"'
            lines += [td(sub_cells.get(m), attrs) for _, sub_cells in subs]
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return lines


def _rank_legend_html():
    """Colour legend of the 1st / 2nd / second to last / last cells."""
    items = [("first", "1st"), ("second", "2nd"), ("before_last", "Second to last"), ("last", "Last")]
    return (
        '<div style="display:flex;gap:16px;align-items:center;font-size:12px;margin:8px 0;">'
        + "".join(
            '<span style="display:inline-flex;align-items:center;gap:4px;">'
            f'<span style="width:12px;height:12px;background:{RANK_COLORS[key]};'
            f'border:1px solid #ccc;border-radius:2px;"></span>{label}</span>'
            for key, label in items)
        + '</div>'
    )


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

    column_tasks = data["column_tasks"]
    columns = []
    for task in column_tasks:
        members = data["group_members"].get(task)
        # Sub-columns: languages for ASR, datasets for others, tasks for the "Others" group
        parts = [(label, data["task_subcol_scores"][task].get(label, {}),
                  data["task_subcol_metric"][task][label])
                 for label in data["task_subcols"].get(task, [])]
        if len(parts) < 2 and not members:
            parts = []
        if members:
            header = task
            note = f"Mean across {len(members)} tasks"
            cells = {m: (f"{v:.2f}", "\n".join([f"{m}\n" + sym_token("R", v.id, "", "p") + note,
                                                *_breakdown(m, parts)]))
                     for m, v in data["task_disp_score"][task].items()}
        else:
            metric = data["task_metric"][task]
            anchor = "cat-" + _slug("Tasks \u00b7 " + task)
            header = _two_line_label(f'<a href="#{anchor}">{task}</a> ({_metric_label(metric)})')
            cells = {m: _score_cell(v, metric, m, _breakdown(m, parts))
                     for m, v in data["task_model_score"][task].items()}
        columns.append((task, header, cells, _sub_columns(parts)))

    lines = [_rank_legend_html()]
    lines += _render_table(table_id, data["all_models"], table_aggregates, columns)
    lines.append(_agg_payload_html(table_id, table_aggregates,
                                   {t: data["task_disp_score"].get(t, {}) for t in column_tasks},
                                   {t: data["task_ascending"].get(t, False) for t in column_tasks}))

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

    # lang -> model -> score
    lang_model_score = defaultdict(dict)
    for e in agg_metric:
        lang_model_score[e["dataset_name"]][e["model_name"]] = e["score"]

    # Per-dataset breakdown: group -> dataset -> model -> score
    lang_ds_model = defaultdict(lambda: defaultdict(dict))
    lang_datasets = defaultdict(set)
    for e in raw_metric:
        lang = group_key_fn(e) if group_key_fn else (e["language"] or "UNKNOWN").upper()
        ds_display = _dataset_display_name(e)
        lang_ds_model[lang][ds_display][e["model_name"]] = e["score"]
        lang_datasets[lang].add(ds_display)
    lang_datasets = {l: sorted(ds) for l, ds in lang_datasets.items()}

    # "Average" column. Languages excluded from the task's average (e.g. Arabic
    # ASR) keep their own column.
    avg_languages = [
        l for l in languages
        if not _excluded_from_task_avg({"task": task, "language": l})
    ]
    average = {}
    for m in all_models:
        scores = [lang_model_score[l][m] for l in avg_languages if m in lang_model_score[l]]
        if scores:
            average[m] = (f"{_display_score(sum(scores) / len(scores), metric):.2f}", "")
    columns = [("Average", "Average", average, [])]

    # Flat mode: show language-prefixed datasets directly (no [+] expand) when
    # either there are few languages (<=2) or few datasets per language (<=2).
    max_ds_per_lang = max((len(lang_datasets.get(l, [])) for l in languages), default=0)
    if len(languages) <= 2 or max_ds_per_lang <= 2:
        for lang in languages:
            for ds in lang_datasets.get(lang, []):
                cells = {m: _score_cell(v, metric, m) for m, v in lang_ds_model[lang][ds].items()}
                columns.append((ds, f"{lang} \u00b7 {ds}", cells, []))
    else:
        for lang in languages:
            ds_list = lang_datasets.get(lang, [])
            parts = [(ds, lang_ds_model[lang][ds], metric) for ds in ds_list] if len(ds_list) >= 2 else []
            header = f"{lang} - {ds_list[0]}" if len(ds_list) == 1 else lang
            cells = {m: _score_cell(v, metric, m, _breakdown(m, parts))
                     for m, v in lang_model_score[lang].items()}
            columns.append((lang, header, cells, _sub_columns(parts)))

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
    lines += _render_table(tbl_id, all_models, table_aggregates, columns)
    ascending = _sort_ascending(metric)
    lines.append(_agg_payload_html(
        tbl_id, table_aggregates,
        {lang: {m: _display_score(v, metric) for m, v in lang_model_score[lang].items()}
         for lang in languages},
        {lang: ascending for lang in languages},
        {lang: dict(lang_model_score[lang]) for lang in languages}))

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

    cat_name = category_override or ("Tasks \u00b7 " + task)
    collector.append({
        "category": cat_name,
        "chart_type": "table",
        "metric": metric,
        "raw_html": "\n".join(lines),
    })


# ---------------------------------------------------------------------------
# HTML Report Builder
# ---------------------------------------------------------------------------

# The page skeleton, its stylesheet and its scripts live in leaderboard/ as plain
# files; build_html_report() inlines them so the report stays a single file.
_LEADERBOARD_DIR = Path(__file__).parent / "leaderboard"
_REPORT_SCRIPTS = ["toggles.js", "tooltip.js", "experiment_filter.js", "table_engine.js", "png_export.js"]


def _render_template():
    """Return the page skeleton with the stylesheet and scripts inlined."""
    template = (_LEADERBOARD_DIR / "template.html").read_text(encoding="utf-8")
    css = (_LEADERBOARD_DIR / "leaderboard.css").read_text(encoding="utf-8")
    scripts = [(_LEADERBOARD_DIR / "js" / name).read_text(encoding="utf-8")
               for name in _REPORT_SCRIPTS]
    template = template.replace("__STYLES__", f"<style>\n{css}</style>")
    return template.replace(
        "__SCRIPTS__", "\n".join(f"<script>\n{js}</script>" for js in scripts))


def _slug(text):
    """Turn a category name into a URL-safe anchor id."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', text).strip('_')


def build_html_report(collected_figures, output_path, default_off_datasets=()):
    """Assemble a single HTML report from collected Plotly figures.

    Figures are grouped by category, with violin plots shown before tables
    within each group.  The sidebar is split into **Overview** and **Tasks**
    groups.

    Symbolic table cells are resolved into data-* attributes and the score
    graph is embedded for the dataset filter; *default_off_datasets* lists the
    dataset indices unchecked when the page loads.
    """
    # Group figures by category, preserving insertion order
    categories = {}
    for item in collected_figures:
        cat = item["category"]
        categories.setdefault(cat, []).append(item)

    # --- Classify categories into overview and tasks ---
    _TASKS_PREFIX = "Tasks \u00b7 "
    overview_cats = []          # category_name
    tasks_cats = []             # (task_label, category_name)

    for cat in categories:
        if cat.startswith("Overview"):
            overview_cats.append(cat)
        elif cat.startswith(_TASKS_PREFIX):
            task = cat[len(_TASKS_PREFIX):]
            tasks_cats.append((task, cat))

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
                    raw = _resolve_sym_tokens(it["raw_html"])
                    section_html += f'    <div class="figure-wrapper">{raw}</div>\n'
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

    html = _render_template().replace('__NAV_ITEMS__', '\n'.join(nav_lines))
    html = html.replace('__SECTIONS__', '\n'.join(section_blocks))
    report_data = json.dumps({
        "nodes": _SYM_NODES,
        "datasets": [list(k) for k in _SYM_DATASETS],
        "super_cats": [_super_category(k[0]) for k in _SYM_DATASETS],
        "off": list(default_off_datasets),
        "colors": {**RANK_COLORS, "missing": MISSING_COLOR},
    }, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace('__REPORT_DATA__', report_data)

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
    parser.add_argument("--config", type=str, default=str(_LEADERBOARD_CONFIG),
                        help="Leaderboard config (datasets, models, names, sizes)")
    parser.add_argument("--violin", action="store_true", help="Include violin plots in the report")
    parser.add_argument(
        "--show-all", "--show_all", dest="show_all", action="store_true",
        help="Bypass BOTH curated filters (ignored datasets + model "
             "allowlist/ignore patterns). Equivalent to "
             "--show_all_models --show_all_datasets.",
    )
    parser.add_argument(
        "--show_all_models", "--show-all-models", dest="show_all_models",
        action="store_true",
        help="Bypass the model allowlist/ignore patterns (show all models). "
             "Ignored datasets are still filtered out.",
    )
    parser.add_argument(
        "--show_all_datasets", "--show-all-datasets", dest="show_all_datasets",
        action="store_true",
        help="Bypass _IGNORED_DATASETS (show ablation datasets). "
             "The model allowlist/ignore patterns still apply.",
    )
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
    use_leaderboard_config(args.config)

    # --show-all is a shorthand that enables both granular flags.
    show_all_models = args.show_all or args.show_all_models
    show_all_datasets = args.show_all or args.show_all_datasets

    # Load all scores. Every dataset is loaded; _IGNORED_DATASETS are only
    # unchecked by default in the report's dataset filter (unless
    # --show_all_datasets), so they can be toggled back on in the browser.
    all_entries = load_all_scores(
        args.input_folder,
        show_all_models=show_all_models,
        show_all_datasets=True,
    )
    if not all_entries:
        print(f"No score files found in {args.input_folder}")
        return

    print(f"Loaded {len(all_entries)} score entries from {len({e['model_name'] for e in all_entries})} models")

    def _default_on(e):
        return show_all_datasets or e["dataset_name"] not in _IGNORED_DATASETS

    # Plotly figures are static: they use the default dataset selection.
    plot_entries = [e for e in all_entries if _default_on(e)]
    # Tables use symbolic scores so the report JS can recompute them.
    entries = _symbolize_entries(all_entries)
    default_off = sorted({
        _SYM_DATASET_IDX[(_task_display_name(e.get("task") or ""),
                          (e.get("language") or "UNKNOWN").upper(),
                          e["dataset_name"])]
        for e in all_entries if not _default_on(e)
    })

    collector = []

    # Overview-only: drop Math QA entries. Math QA is EN-only; keeping it as a
    # sibling task of QUESTION ANSWERING under the QA super-category gave it
    # ~50% weight in the QA aggregate. It still shows up in the per-task QA
    # summary section below, which uses the raw `entries`.
    def _not_math_qa(e):
        return e.get("task", "").upper() != "MATH QUESTION ANSWERING"

    overview_entries = [e for e in entries if _not_math_qa(e)]
    overview_plot_entries = [e for e in plot_entries if _not_math_qa(e)]

    # --- Step 0: Overview table (all tasks × models) ---
    plot_overview_table(overview_entries, collector,
                        table_aggregates=args.table_aggregates)
    plot_size_vs_performance(overview_plot_entries, collector,
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

    def _fren_core(e):
        return _lang_match(e) and _super_category(e.get("task", "")) in _allowed_sc

    filtered = [e for e in overview_entries if _fren_core(e)]
    if filtered:
        plot_overview_table(
            filtered, collector,
            title="Overview (FR/EN \u2014 ASR, AST, QA)",
            table_id="overview-filtered-tbl",
            allowed_super_cats=_allowed_sc,
            table_aggregates=args.table_aggregates,
        )
        filtered_plot = [e for e in overview_plot_entries if _fren_core(e)]
        if filtered_plot:
            plot_size_vs_performance(
                filtered_plot, collector,
                category="Overview (FR/EN \u2014 ASR, AST, QA)",
                allowed_super_cats=_allowed_sc,
                figure_aggregates=args.figure_aggregates,
            )

    # --- Steps 1+2: Super-category sections (violin plots + summary tables) ---
    for super_cat, task_map in _group_by_super_category(entries).items():
        cat_label = f"Tasks \u00b7 {super_cat}"
        multi_task = len(task_map) > 1

        # Violin plots: one per task within the super-category
        if args.violin:
            for task in sorted(task_map):
                task_plot = [e for e in plot_entries
                             if _task_display_name(e.get("task") or "") == task
                             and _super_category(e.get("task", "")) == super_cat]
                if task_plot:
                    plot_violin_charts(task_plot, cat_label, collector)

        # Summary tables per task within the super-category
        for task, task_raw in sorted(task_map.items()):
            subtitle = task if multi_task else None
            plot_summary_tables(task_raw, collector,
                                category_override=cat_label,
                                subtitle=subtitle,
                                table_aggregates=args.table_aggregates)

    if not collector:
        print("No figures generated.")
        return

    output_path = os.path.join(args.output_folder, "index.html")
    build_html_report(collector, output_path, default_off_datasets=default_off)


if __name__ == "__main__":
    main()
