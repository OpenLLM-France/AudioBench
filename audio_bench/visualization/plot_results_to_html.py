#!/usr/bin/env python3
"""Plot AudioBench evaluation results as a single interactive HTML report.

Generates one HTML file with an Overview table (all tasks × models ranked)
and per-task Summary sections (language-column tables with expandable
per-dataset sub-columns, and optionally violin plots).

Output: {output_folder}/report.html

Usage examples:
    python -m audio_bench.visualization.plot_results_to_html results/
    python -m audio_bench.visualization.plot_results_to_html results/ --violin
    python -m audio_bench.visualization.plot_results_to_html results/ --output_folder my_plots/
"""

import argparse
import html
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

_IGNORED_DATASETS = {
    "StressTest_SSR",
    "VoxCeleb-accent",
    "VoxCeleb",  # Speaker identification — only run on some models
    "MuChoMusic",
    "SLUE-SQA5_format_json_answer", # Format following
    "SLUE-SQA5_time2sentence", "SLUE-SQA5_time2word", "SLUE-SQA5_word2sentence", "SLUE-SQA5_word2time", # Information Extraction
    "SLUE-SQA5_format_timestamped_transcription", # Timestamped transcription
}

# (task, language) pairs excluded from a task's AVERAGE (still shown individually
# as their own sub-column/row — just not folded into the super-category mean).
# Arabic ASR is dropped from the ASR average but stays visible as its own AR column.
_AVG_EXCLUDED_TASK_LANGS = {("ASR", "AR")}


def _excluded_from_task_avg(entry):
    """True if this entry's (task, language) must be dropped from task averages."""
    task = str(entry.get("task") or "").upper()
    lang = str(entry.get("language") or "").upper()
    return (task, lang) in _AVG_EXCLUDED_TASK_LANGS

# CONSORTIUM_NAME = "LINAGORA"
CONSORTIUM_NAME = "OpenLLM-France"

ONLY_SHOW_CONSORTIUM_MODELS = {
    "LINAGORA/Canary_Luciole-1B-SFT-1.1_v3_buckets_s082829": f"{CONSORTIUM_NAME}/Canary_Luciole-1B",
    "LINAGORA/Canary_Qwen3-1.7B_v2_buckets_s011803": f"{CONSORTIUM_NAME}/Canary_Qwen3-1.7B",
}

_IGNORED_MODEL_PATTERNS = [re.compile(p) for p in [
    r"^LINAGORA/Canary_Qwen3-1\.7B_v1.*",
    r".*xp_timestamp.*",
    # r"^.*/(?!.*data-v1).*",
]]

def _is_model_ignored(model_name):
    if ONLY_SHOW_CONSORTIUM_MODELS and CONSORTIUM_NAME in model_name:
        return model_name not in ONLY_SHOW_CONSORTIUM_MODELS.values()
    return any(p.search(model_name) for p in _IGNORED_MODEL_PATTERNS)

_MODEL_NAME_CORRECTIONS = {
    "LINAGORA/Canary-Qwen3-5B-Thinking": f"{CONSORTIUM_NAME}/Canary-Qwen3-4B_data-v1_8h",
    "LINAGORA/Canary-Qwen3-1.7B-v2": f"{CONSORTIUM_NAME}/Canary-Qwen3-1.7B_data-v1_8h",
    "LINAGORA/Canary_Luciole-1B-SFT-1.1_v3_buckets_s010000": f"{CONSORTIUM_NAME}/Canary_Luciole-1B_LLM-LoRA_8h",
    "LINAGORA/Canary_Luciole-1B-SFT-1.1_v3_buckets_s082829": f"{CONSORTIUM_NAME}/Canary_Luciole-1B_LLM-LoRA_40h",
    "LINAGORA/Canary_Luciole-1B-SFT-1.1_v2_s027458": f"{CONSORTIUM_NAME}/Canary_Luciole-1B_LLM-LoRA-no_bucket_8h",
    "LINAGORA/Canary_Qwen3-1.7B_v2_buckets_s011803": f"{CONSORTIUM_NAME}/Canary_Qwen3-1.7B_LLM-LoRA_8h",
    "LINAGORA/Canary_Luciole-1B-SFT-1.1_v4_adapter_s005000": f"{CONSORTIUM_NAME}/Canary_Luciole-1B_Step-Adapter_2h",
    "LINAGORA/Canary_Luciole-1B-SFT-1.1_v4_encoder_s020000": f"{CONSORTIUM_NAME}/Canary_Luciole-1B_Step-Encoder_8h",
    "LINAGORA/Canary_Luciole-1B-SFT-1.1_v4_encoder_s217983": f"{CONSORTIUM_NAME}/Canary_Luciole-1B_Step-Encoder_80h",
} | ONLY_SHOW_CONSORTIUM_MODELS

# Maps the (possibly shortened) display model name -> the model id (the
# results folder name). Populated by load_all_scores and used to show the
# model id as a hover tooltip in the tables.
_MODEL_CHECKPOINT = {}


def _model_name_td(m):
    """Render the model-name table cell, showing the model id on hover."""
    model_id = _MODEL_CHECKPOINT.get(m, m)
    title_attr = f' title="{html.escape(model_id, quote=True)}"' if model_id != m else ""
    return f'<td class="mname"{title_attr}>{m}</td>'

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


# Patterns are tried in order; first regex match wins. Exact strings work too
# since they are compiled as regexes.
_MODEL_SIZE_OVERRIDES = [
    (rf"^{CONSORTIUM_NAME}/Canary[-_]Qwen3[-_]1\.7B", "2.5B"),
    (rf"^{CONSORTIUM_NAME}/Canary[-_]Qwen3[-_]4B", "4.8B"),
    (rf"^{CONSORTIUM_NAME}/Canary_Luciole-1B", "2.1B"),
    (r"^LINAGORA/luciole_v\d+_stage1_canary", "2.1B"),
    (r"^LINAGORA/luciole_v\d+", "1.9B"),
    (r"^LINAGORA/luciole8b_", "8.8B"),
    (r"^LINAGORA/luciole23b_", "24.1B"),
    (r"^microsoft/Phi-4-multimodal-instruct$", "5.6B"),
    (r"^nvidia/audio-flamingo-3-hf$", "8.2B"),
    (r"^Qwen/Qwen2-Audio-7B-Instruct$", "8.4B"),
    (r"^Qwen/Qwen2\.5-Omni-7B$", "11B"),
    (r"^Qwen/Qwen2\.5-Omni-3B$", "5.9B"),
    (r"^Qwen/Qwen3-Omni-30B-A3B-Instruct$", "35.3B"),
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


# Task display order for per-language summary tables.
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
    * ``data-tt`` -- tooltip template, tokens written as ``[[kind:id:args]]``
    Tokens are replaced by their initial text, so the page renders as before.
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
            return (f' title="{plain(text)}"'
                    f' data-tt="{html.escape(template, quote=True)}"')
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


def _compute_ci(std, n):
    """Compute 95% confidence interval half-width, or None."""
    if std is None or n is None or n <= 0:
        return None
    return 1.96 * std / math.sqrt(n)


def _format_score_with_ci(score, metric, std=None, n=None, model=None, rank=None):
    """Return (html_str, tooltip_str) with optional CI display.

    html_str:   '18.50 <span class="ci">±0.32</span>'  (or just '18.50')
    tooltip_str: '18.50 [18.18, 18.82], n=676'          (or just '18.50')

    When *model* is given, it is prepended to the tooltip (helps identify the
    row on wide tables where the model column has scrolled out of view).
    When *rank* is a (position, total) tuple, the column ranking is inserted
    (e.g. '3e').
    """
    disp = _display_score(score, metric)
    sym = isinstance(disp, SymScore)
    pct = int(metric in ZERO_TO_ONE_RANGE)
    base = f"{disp:.2f}"
    ci = _compute_ci(std, n)
    if ci is not None:
        # Scale CI the same way as the score display
        ci_disp = ci * 100 if metric in ZERO_TO_ONE_RANGE else ci
        lo = float(disp) - ci_disp
        hi = float(disp) + ci_disp
        html_str = f'{base} <span class="ci">\u00b1{ci_disp:.2f}</span>'
        ci_str = f" [{lo:.2f}, {hi:.2f}], n={n}"
    else:
        html_str = base
        ci_str = ""
    if sym:
        # The JS recomputes the CI (when available) and marks the cell as CI-capable.
        html_str = base + sym_token("CI", disp.id, html_str[len(base):], pct)
        ci_str = sym_token("C", disp.id, ci_str, pct, "m")
    tooltip_str = base + ci_str
    if rank:
        rank_str = f"{rank[0]}e — "
        tooltip_str = (sym_token("R", disp.id, rank_str, "p") if sym else rank_str) + tooltip_str
    if model:
        tooltip_str = f"{model}\n{tooltip_str}"
    return html_str, tooltip_str


def _tooltip_subline(name, score, metric, std=None, n=None, rank=None):
    """One sub-item tooltip line, e.g. 'FLEURS: 18.50±1.20 (3e)'.

    Used to enumerate the datasets/languages hidden behind an expandable cell.
    """
    disp = _display_score(score, metric)
    ci = _compute_ci(std, n)
    ci_str = ""
    if ci is not None:
        ci_disp = ci * 100 if metric in ZERO_TO_ONE_RANGE else ci
        ci_str = f"±{ci_disp:.2f}"
    rank_str = f" ({rank[0]}e)" if rank else ""
    if isinstance(disp, SymScore):
        ci_str = sym_token("C", disp.id, ci_str, int(metric in ZERO_TO_ONE_RANGE), "s")
        rank_str = sym_token("R", disp.id, rank_str, "s")
    return f"{name}: {disp:.2f}{ci_str}{rank_str}"


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


def _full_ranks(pairs, asc):
    """Return {row_index: (rank, total)} for every row in *pairs* (1 = best).

    *pairs* is a list of (value, row_index). *asc* True means lower is better.
    Ties are broken positionally (stable sort), so ranks are always 1..N.
    """
    if not pairs:
        return {}
    ranked = sorted(pairs, key=lambda x: x[0], reverse=not asc)
    total = len(ranked)
    return {ri: (i + 1, total) for i, (_, ri) in enumerate(ranked)}


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
    others_tasks = sc_to_tasks.get("Others", [])

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
        task_lang_scores[task][lang][m] = (e["score"], e.get("std"), e.get("n"))
        task_languages[task].add(lang)
    task_languages = {t: sorted(langs, key=_lang_sort_key) for t, langs in task_languages.items()}

    # --- Per-task sub-columns ---
    # ASR: sub-columns = languages (avg across datasets per language)
    # Others: sub-columns = lang-prefixed datasets
    task_subcols = {}       # task -> list of sub-column display names
    task_subcol_scores = {} # task -> subcol -> model -> (score, std, n)
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
                        score = sum(e["score"] for e in ents) / len(ents)
                        pooled = []
                        for e in ents:
                            if "all_scores" in e:
                                pooled.extend(e["all_scores"])
                        if pooled:
                            subcol_scores[st][m] = (score, float(np.std(np.array(pooled))), len(pooled))
                        else:
                            subcol_scores[st][m] = (score, None, None)
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
                        score = sum(e["score"] for e in ds_entries) / len(ds_entries)
                        pooled = []
                        for e in ds_entries:
                            if "all_scores" in e:
                                pooled.extend(e["all_scores"])
                        if pooled:
                            subcol_scores[display][m] = (score, float(np.std(np.array(pooled))), len(pooled))
                        else:
                            subcol_scores[display][m] = (score, None, None)
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
        for m, (val, *_) in task_model_score.get(task, {}).items():
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
                unit = " %" if metric in ZERO_TO_ONE_RANGE else ""
                label = f"{t} ({metric.upper()}{unit})"
                subcols.append(label)
                scores_map = {}
                for m, (val, st, n) in task_model_score.get(t, {}).items():
                    scores_map[m] = (val, st, n)
                subcol_scores[label] = scores_map
                subcol_metric[label] = metric
        elif group_label == "QA":
            # Per-language sub-columns: average across member tasks.
            lang_to_model_vals = defaultdict(lambda: defaultdict(list))
            for t in members:
                for lang, model_map in task_lang_scores.get(t, {}).items():
                    for mdl, score_tuple in model_map.items():
                        if score_tuple and score_tuple[0] is not None:
                            lang_to_model_vals[lang][mdl].append(score_tuple[0])
            subcols = sorted(lang_to_model_vals.keys(), key=_lang_sort_key)
            subcol_scores = {}
            subcol_metric = {}
            metric_counts = defaultdict(int)
            for t in members:
                metric_counts[task_metric[t]] += 1
            group_metric = max(metric_counts, key=metric_counts.get) if metric_counts else None
            for lang in subcols:
                subcol_scores[lang] = {
                    mdl: (sum(vals) / len(vals), None, None)
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
        "agg": agg,
        "task_entries": task_entries,
        "task_metric": task_metric,
        "tasks": tasks,
        "all_models": all_models,
        "task_model_score": task_model_score,
        "ordered_tasks": ordered_tasks,
        "column_tasks": column_tasks,
        "others_tasks": others_tasks,
        "group_members": group_members,
        "task_ascending": task_ascending,
        "task_disp_score": task_disp_score,
        "task_subcols": task_subcols,
        "task_subcol_scores": task_subcol_scores,
        "task_subcol_metric": task_subcol_metric,
        "task_languages": task_languages,
        "task_lang_scores": task_lang_scores,
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
    all_models = data["all_models"]
    task_model_score = data["task_model_score"]
    ordered_tasks = data["ordered_tasks"]
    column_tasks = data["column_tasks"]
    others_tasks = data["others_tasks"]
    group_members = data["group_members"]
    task_ascending = data["task_ascending"]
    task_disp_score = data["task_disp_score"]
    task_subcols = data["task_subcols"]
    task_subcol_scores = data["task_subcol_scores"]
    task_subcol_metric = data["task_subcol_metric"]
    task_model_rank = data["task_model_rank"]
    model_avg_rank = data["model_avg_rank"]
    model_minmax = data["model_minmax"]
    model_zscore = data["model_zscore"]
    sorted_models = data["sorted_models"]

    # --- Ranked row indices for highlighting ---
    task_ranks = {}
    task_full_ranks = {}
    for task in column_tasks:
        asc = task_ascending[task]
        pairs = [
            (task_disp_score[task][m], ri)
            for ri, m in enumerate(sorted_models) if m in task_disp_score[task]
        ]
        task_ranks[task] = _ranked_rows(pairs, asc)
        task_full_ranks[task] = _full_ranks(pairs, asc)

    # Ranked rows for sub-columns (languages for ASR, datasets for others, tasks for Others group)
    task_subcol_ranks = defaultdict(dict)
    task_subcol_full_ranks = defaultdict(dict)
    for task in column_tasks:
        for subcol in task_subcols.get(task, []):
            metric = task_subcol_metric[task][subcol]
            sub_asc = _sort_ascending(metric)
            scores_map = task_subcol_scores[task].get(subcol, {})
            pairs = [
                (_display_score(scores_map[m][0], metric), ri)
                for ri, m in enumerate(sorted_models) if m in scores_map
            ]
            task_subcol_ranks[task][subcol] = _ranked_rows(pairs, sub_asc)
            task_subcol_full_ranks[task][subcol] = _full_ranks(pairs, sub_asc)

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
    top, bottom = [], []
    top.append('<th rowspan="2">Model</th>')
    top.append('<th rowspan="2">Size</th>')
    top.append(f'<th colspan="{len(table_aggregates)}">Aggregation</th>')
    for agg_name in table_aggregates:
        bottom.append(f"<th>{_AGG_META[agg_name]['label']}</th>")
    for task in column_tasks:
        task_slug = _slug(task)
        is_group = task in group_members
        if is_group:
            label = task
        else:
            cat_label = "Tasks \u00b7 " + task
            section_anchor = f"cat-{_slug(cat_label)}"
            metric = task_metric[task]
            unit = " %" if metric in ZERO_TO_ONE_RANGE else ""
            label = _two_line_label(
                f'<a href="#{section_anchor}">{task}</a> ({metric.upper()}{unit})'
            )
        subcols = task_subcols.get(task, [])
        if len(subcols) >= 2 or is_group:
            top.append(
                f'<th rowspan="2">{label} '
                f'<button class="toggle-btn" onclick="toggleOvTask(this,\'{task_slug}\')">+</button></th>'
            )
            for subcol in subcols:
                top.append(
                    f'<th rowspan="2" class="lang-col" data-task="{task_slug}">{subcol}</th>'
                )
        else:
            top.append(f'<th rowspan="2">{label}</th>')

    lines.append("<thead><tr>" + "".join(top) + "</tr><tr>" + "".join(bottom) + "</tr></thead>")

    # --- Body ---
    lines.append("<tbody>")
    for ri, m in enumerate(sorted_models):
        lines.append(f'<tr data-model="{html.escape(m, quote=True)}">')
        lines.append(_model_name_td(m))
        lines.append(f"<td>{_extract_model_size(m)}</td>")

        # Aggregate columns
        for agg_name in table_aggregates:
            ar = agg_render[agg_name]
            v = ar["values"][m]
            agg_attr = f' data-agg="{agg_name}"'
            if v != ar["sentinel"]:
                lines.append(_td(f"{v:{ar['fmt']}}", rank_key=ar["ranks"].get(ri),
                                 extra_attrs=agg_attr))
            else:
                lines.append(_td("-", is_missing=True, extra_attrs=agg_attr))

        for task in column_tasks:
            task_slug = _slug(task)
            subcols = task_subcols.get(task, [])
            is_group = task in group_members
            expandable = len(subcols) >= 2 or is_group

            # When the cell can be expanded, list its sub-items in the tooltip.
            sub_suffix = ""
            if expandable:
                sub_lines = []
                for subcol in subcols:
                    smap = task_subcol_scores[task].get(subcol, {})
                    if m in smap:
                        sv0, sst0, sn0 = smap[m]
                        sub_lines.append(_tooltip_subline(
                            subcol, sv0, task_subcol_metric[task][subcol], sst0, sn0,
                            rank=task_subcol_full_ranks[task].get(subcol, {}).get(ri)))
                if sub_lines:
                    sub_suffix = "\n" + "\n".join(sub_lines)

            # Main task cell
            if is_group:
                members = group_members[task]
                if m in task_disp_score.get(task, {}):
                    val = task_disp_score[task][m]
                    rk = task_full_ranks.get(task, {}).get(ri)
                    rank_str = f"{rk[0]}e — " if rk else ""
                    if isinstance(val, SymScore):
                        rank_str = sym_token("R", val.id, rank_str, "p")
                    lines.append(_td(f"{val:.2f}",
                                     rank_key=task_ranks.get(task, {}).get(ri),
                                     title=f"{m}\n{rank_str}Mean across {len(members)} tasks{sub_suffix}"))
                else:
                    lines.append(_td("-", is_missing=True))
            else:
                metric = task_metric[task]
                if m in task_model_score.get(task, {}):
                    t_val, st, n = task_model_score[task][m]
                    html_v, tip = _format_score_with_ci(
                        t_val, metric, st, n, model=m,
                        rank=task_full_ranks.get(task, {}).get(ri))
                    lines.append(_td(html_v, rank_key=task_ranks.get(task, {}).get(ri),
                                     title=tip + sub_suffix))
                else:
                    lines.append(_td("-", is_missing=True))

            # Sub-column cells (hidden by default)
            if expandable:
                for subcol in subcols:
                    attr = f' class="lang-col" data-task="{task_slug}"'
                    sub_metric = task_subcol_metric[task][subcol]
                    scores_map = task_subcol_scores[task].get(subcol, {})
                    if m in scores_map:
                        sv, st, n = scores_map[m]
                        html_v, tip = _format_score_with_ci(
                            sv, sub_metric, st, n, model=m,
                            rank=task_subcol_full_ranks[task].get(subcol, {}).get(ri))
                        lines.append(_td(html_v,
                                         rank_key=task_subcol_ranks[task].get(subcol, {}).get(ri),
                                         extra_attrs=attr, title=tip))
                    else:
                        lines.append(_td("-", is_missing=True, extra_attrs=attr))

        lines.append("</tr>")

    lines.append("</tbody></table>")
    lines.append(_agg_payload_html(table_id, table_aggregates,
                                   {t: task_disp_score.get(t, {}) for t in column_tasks},
                                   {t: task_ascending.get(t, False) for t in column_tasks}))

    # JavaScript toggle
    lines.append("""\
<script>
function toggleOvTask(btn, task) {
  var tbl = btn.closest('table');
  var cells = tbl.querySelectorAll('[data-task="' + task + '"]');
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

    # Flat mode: show language-prefixed datasets directly (no [+] expand) when
    # either there are few languages (<=2) or few datasets per language (<=2).
    max_ds_per_lang = max((len(lang_datasets.get(l, [])) for l in languages), default=0)
    use_flat_mode = (len(languages) <= 2 or max_ds_per_lang <= 2) and len(languages) >= 1
    # Build flat columns as (lang, ds) pairs if using flat mode
    flat_cols = []
    if use_flat_mode:
        for lang in languages:
            for ds in lang_datasets.get(lang, []):
                flat_cols.append((lang, ds))

    # Average per model across languages
    ascending = _sort_ascending(metric)
    # Compute model_avg (needed for the "Average" column). Drop languages excluded
    # from this task's average (e.g. Arabic ASR) — they keep their own column.
    avg_languages = [
        l for l in languages
        if not _excluded_from_task_avg({"task": task, "language": l})
    ]
    model_avg = {}
    for m in all_models:
        scores = [lang_model_score[l][m][0] for l in avg_languages if m in lang_model_score[l]]
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
    lang_full_ranks = {}
    for lang in languages:
        pairs = [
            (_display_score(lang_model_score[lang][m][0], metric), ri)
            for ri, m in enumerate(sorted_models) if m in lang_model_score[lang]
        ]
        lang_ranks[lang] = _ranked_rows(pairs, ascending)
        lang_full_ranks[lang] = _full_ranks(pairs, ascending)

    lang_ds_ranks = defaultdict(dict)
    lang_ds_full_ranks = defaultdict(dict)
    rank_langs = flat_cols and [l for l, _ in flat_cols] or expandable_langs
    for lang in (set(rank_langs) if use_flat_mode else expandable_langs):
        for ds in lang_datasets.get(lang, []):
            pairs = [
                (_display_score(lang_ds_model[lang][ds][m][0], metric), ri)
                for ri, m in enumerate(sorted_models) if m in lang_ds_model[lang][ds]
            ]
            lang_ds_ranks[lang][ds] = _ranked_rows(pairs, ascending)
            lang_ds_full_ranks[lang][ds] = _full_ranks(pairs, ascending)

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
    top, bottom = [], []
    top.append('<th rowspan="2">Model</th>')
    top.append('<th rowspan="2">Size</th>')
    top.append(f'<th colspan="{len(table_aggregates)}">Aggregation</th>')
    for agg_name in table_aggregates:
        bottom.append(f"<th>{_AGG_META[agg_name]['label']}</th>")
    top.append('<th rowspan="2">Average</th>')
    if use_flat_mode:
        for lang, ds in flat_cols:
            top.append(f'<th rowspan="2">{lang} \u00b7 {ds}</th>')
    else:
        for lang in languages:
            grp = _slug(lang)
            if lang in expandable_langs:
                top.append(
                    f'<th rowspan="2">{lang} '
                    f'<button class="toggle-btn" onclick="toggleCols(this,\'{tbl_id}\',\'{grp}\')">+</button></th>'
                )
                for ds in lang_datasets[lang]:
                    top.append(f'<th rowspan="2" class="lang-col" data-group="{grp}">{ds}</th>')
            else:
                ds_list = lang_datasets.get(lang, [])
                if len(ds_list) == 1:
                    top.append(f'<th rowspan="2">{lang} - {ds_list[0]}</th>')
                else:
                    top.append(f'<th rowspan="2">{lang}</th>')
    lines.append("<thead><tr>" + "".join(top) + "</tr><tr>" + "".join(bottom) + "</tr></thead>")

    # Body
    lines.append("<tbody>")
    for ri, m in enumerate(sorted_models):
        lines.append(f'<tr data-model="{html.escape(m, quote=True)}">')
        lines.append(_model_name_td(m))
        lines.append(f"<td>{_extract_model_size(m)}</td>")

        # Aggregate columns
        for agg_name in table_aggregates:
            ar = agg_render[agg_name]
            v = ar["values"][m]
            agg_attr = f' data-agg="{agg_name}"'
            if v != ar["sentinel"]:
                lines.append(_td(f"{v:{ar['fmt']}}", rank_key=ar["ranks"].get(ri),
                                 extra_attrs=agg_attr))
            else:
                lines.append(_td("-", is_missing=True, extra_attrs=agg_attr))

        # Average
        if model_avg[m] is not None:
            val = f"{_display_score(model_avg[m], metric):.2f}"
            lines.append(_td(val, rank_key=avg_ranks.get(ri)))
        else:
            lines.append(_td("-", is_missing=True))

        if use_flat_mode:
            for lang, ds in flat_cols:
                if m in lang_ds_model[lang][ds]:
                    sc, st, n = lang_ds_model[lang][ds][m]
                    html_v, tip = _format_score_with_ci(
                        sc, metric, st, n, model=m,
                        rank=lang_ds_full_ranks[lang].get(ds, {}).get(ri))
                    lines.append(_td(html_v,
                                     rank_key=lang_ds_ranks[lang].get(ds, {}).get(ri),
                                     title=tip))
                else:
                    lines.append(_td("-", is_missing=True))
        else:
            for lang in languages:
                grp = _slug(lang)

                # Aggregate cell
                if m in lang_model_score[lang]:
                    sc, st, n = lang_model_score[lang][m]
                    html_v, tip = _format_score_with_ci(
                        sc, metric, st, n, model=m,
                        rank=lang_full_ranks.get(lang, {}).get(ri))
                    if lang in expandable_langs:
                        sub_lines = []
                        for ds in lang_datasets.get(lang, []):
                            if m in lang_ds_model[lang][ds]:
                                dsc, dst, dn = lang_ds_model[lang][ds][m]
                                sub_lines.append(_tooltip_subline(
                                    ds, dsc, metric, dst, dn,
                                    rank=lang_ds_full_ranks[lang].get(ds, {}).get(ri)))
                        if sub_lines:
                            tip = tip + "\n" + "\n".join(sub_lines)
                    lines.append(_td(html_v, rank_key=lang_ranks.get(lang, {}).get(ri), title=tip))
                else:
                    lines.append(_td("-", is_missing=True))

                # Per-dataset sub-cells
                if lang in expandable_langs:
                    for ds in lang_datasets[lang]:
                        attr = f' class="lang-col" data-group="{grp}"'
                        if m in lang_ds_model[lang][ds]:
                            sc, st, n = lang_ds_model[lang][ds][m]
                            html_v, tip = _format_score_with_ci(
                                sc, metric, st, n, model=m,
                                rank=lang_ds_full_ranks[lang].get(ds, {}).get(ri))
                            lines.append(_td(html_v, rank_key=lang_ds_ranks[lang].get(ds, {}).get(ri),
                                             extra_attrs=attr, title=tip))
                        else:
                            lines.append(_td("-", is_missing=True, extra_attrs=attr))

        lines.append("</tr>")

    lines.append("</tbody></table>")
    lines.append(_agg_payload_html(
        tbl_id, table_aggregates, lang_disp_scores,
        {lang: ascending for lang in languages},
        {lang: {m: v[0] for m, v in lang_model_score[lang].items()} for lang in languages}))

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
                           table_aggregates=None, violin_entries=None):
    """Build per-language-group sections (French, English, Others).

    Each group gets violin plots per task (when *include_violin* is True)
    and a summary table (Models × Tasks) with expandable per-dataset
    sub-columns.  *violin_entries*, when given, replaces *entries* for the
    violin plots (plain float scores).
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
            violin_map = defaultdict(list)
            for e in (violin_entries if violin_entries is not None else grp_ents):
                if e.get("task") and _classify_language(e.get("language")) == group_name:
                    violin_map[e["task"]].append(e)
            for task in sorted(violin_map.keys()):
                plot_violin_charts(violin_map[task], category, collector)

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

    tasks = sorted(task_entries_map.keys(), key=_lang_table_task_sort_key)
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
    expandable_tasks = {t for t in tasks if len(task_datasets.get(t, [])) >= 2}

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
    task_full_ranks = {}
    for task in tasks:
        metric = task_metric[task]
        asc = ascending_map[task]
        pairs = [
            (_display_score(task_model_score[task][m][0], metric), ri)
            for ri, m in enumerate(sorted_models) if m in task_model_score[task]
        ]
        task_ranks[task] = _ranked_rows(pairs, asc)
        task_full_ranks[task] = _full_ranks(pairs, asc)

    task_ds_ranks = defaultdict(dict)
    task_ds_full_ranks = defaultdict(dict)
    for task in expandable_tasks:
        metric = task_metric[task]
        asc = ascending_map[task]
        for ds in task_datasets[task]:
            pairs = [
                (_display_score(task_ds_model[task][ds][m][0], metric), ri)
                for ri, m in enumerate(sorted_models) if m in task_ds_model[task][ds]
            ]
            task_ds_ranks[task][ds] = _ranked_rows(pairs, asc)
            task_ds_full_ranks[task][ds] = _full_ranks(pairs, asc)

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
    top, bottom = [], []
    top.append('<th rowspan="2">Model</th>')
    top.append('<th rowspan="2">Size</th>')
    top.append(f'<th colspan="{len(table_aggregates)}">Aggregation</th>')
    for agg_name in table_aggregates:
        bottom.append(f"<th>{_AGG_META[agg_name]['label']}</th>")
    for task in tasks:
        metric = task_metric[task]
        unit = " %" if metric in ZERO_TO_ONE_RANGE else ""
        slug = _slug(task)
        label = _two_line_label(f'{task} ({metric.upper()}{unit})')
        if task in expandable_tasks:
            top.append(
                f'<th rowspan="2">{label} '
                f'<button class="toggle-btn" onclick="toggleCols(this,\'{tbl_id}\',\'{slug}\')">+</button></th>'
            )
            for ds in task_datasets[task]:
                top.append(f'<th rowspan="2" class="lang-col" data-group="{slug}">{ds}</th>')
        else:
            top.append(f'<th rowspan="2">{label}</th>')
    lines.append("<thead><tr>" + "".join(top) + "</tr><tr>" + "".join(bottom) + "</tr></thead>")

    # Body
    lines.append("<tbody>")
    for ri, m in enumerate(sorted_models):
        # Skip models with no data in this group
        if not any(m in task_model_score[t] for t in tasks):
            continue
        lines.append(f'<tr data-model="{html.escape(m, quote=True)}">')
        lines.append(_model_name_td(m))
        lines.append(f"<td>{_extract_model_size(m)}</td>")

        # Aggregate columns
        for agg_name in table_aggregates:
            ar = agg_render[agg_name]
            v = ar["values"][m]
            agg_attr = f' data-agg="{agg_name}"'
            if v != ar["sentinel"]:
                lines.append(_td(f"{v:{ar['fmt']}}", rank_key=ar["ranks"].get(ri),
                                 extra_attrs=agg_attr))
            else:
                lines.append(_td("-", is_missing=True, extra_attrs=agg_attr))

        for task in tasks:
            metric = task_metric[task]
            slug = _slug(task)

            if m in task_model_score[task]:
                sc, st, n = task_model_score[task][m]
                html_v, tip = _format_score_with_ci(
                    sc, metric, st, n, model=m,
                    rank=task_full_ranks.get(task, {}).get(ri))
                if task in expandable_tasks:
                    sub_lines = []
                    for ds in task_datasets.get(task, []):
                        if m in task_ds_model[task][ds]:
                            dsc, dst, dn = task_ds_model[task][ds][m]
                            sub_lines.append(_tooltip_subline(
                                ds, dsc, metric, dst, dn,
                                rank=task_ds_full_ranks[task].get(ds, {}).get(ri)))
                    if sub_lines:
                        tip = tip + "\n" + "\n".join(sub_lines)
                lines.append(_td(html_v, rank_key=task_ranks.get(task, {}).get(ri), title=tip))
            else:
                lines.append(_td("-", is_missing=True))

            if task in expandable_tasks:
                for ds in task_datasets[task]:
                    attr = f' class="lang-col" data-group="{slug}"'
                    if m in task_ds_model[task][ds]:
                        sc, st, n = task_ds_model[task][ds][m]
                        html_v, tip = _format_score_with_ci(
                            sc, metric, st, n, model=m,
                            rank=task_ds_full_ranks[task].get(ds, {}).get(ri))
                        lines.append(_td(html_v, rank_key=task_ds_ranks[task].get(ds, {}).get(ri),
                                         extra_attrs=attr, title=tip))
                    else:
                        lines.append(_td("-", is_missing=True, extra_attrs=attr))

        lines.append("</tr>")

    lines.append("</tbody></table>")
    lines.append(_agg_payload_html(
        tbl_id, table_aggregates, task_disp_scores, ascending_map,
        {t: {m: v[0] for m, v in task_model_score[t].items()} for t in tasks}))

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
         display: flex; min-height: 100vh; background: #ffffff; color: #222; }

  /* Sidebar */
  nav.sidebar { position: fixed; top: 0; left: 0; width: 360px; height: 100vh;
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

  /* Filter panels (experiments, datasets) */
  .flt-panel { border-top: 1px solid #334155; margin-top: 8px; padding-top: 10px; }
  .flt-panel .xp-filter-head { display: flex; align-items: center; justify-content: space-between;
                                padding: 0 16px 8px; }
  .flt-panel .xp-filter-head span { font-size: 11px; font-weight: 700; text-transform: uppercase;
                                     color: #64748b; letter-spacing: .05em; }
  .flt-panel .xp-filter-head button { font-size: 11px; background: #334155; color: #e2e8f0;
                                     border: 1px solid #475569; border-radius: 4px; padding: 3px 8px;
                                     cursor: pointer; }
  .flt-panel .xp-filter-head button:hover { background: #475569; }
  .flt-tree { max-height: 60vh; overflow-y: auto; padding: 0 10px 10px; }
  .flt-tree ul { list-style: none; margin: 0; padding-left: 16px; }
  .flt-tree > ul { padding-left: 0; }
  .flt-tree li { padding: 0; }
  .flt-tree label { display: flex; align-items: center; gap: 6px; padding: 3px 6px;
                           font-size: 12px; color: #cbd5e1; cursor: pointer; border-radius: 4px;
                           white-space: normal; word-break: break-word; line-height: 1.3; }
  .flt-tree label:hover { background: #334155; color: #e2e8f0; }
  .flt-tree input[type="checkbox"] { flex: none; accent-color: #3b82f6; margin-top: 1px; }
  .flt-tree .xp-group { font-weight: 600; color: #e2e8f0; }
  .flt-tree details { margin: 1px 0; }
  .flt-tree summary { list-style: none; cursor: pointer; display: flex; align-items: center;
                             gap: 6px; padding: 3px 6px; border-radius: 4px; }
  .flt-tree summary::-webkit-details-marker { display: none; }
  .flt-tree summary:hover { background: #334155; }
  .flt-tree summary .xp-caret { flex: none; width: 16px; text-align: center;
                                       font-size: 16px; font-weight: 700; color: #e2e8f0;
                                       transition: transform .1s; }
  .flt-tree details[open] > summary .xp-caret { transform: rotate(90deg); }
  .flt-tree summary .xp-count { flex: none; font-size: 10px; color: #64748b; font-weight: 400; }
  .flt-panel .xp-filter-head .flt-btns { display: flex; gap: 4px; }
  tr.xp-hidden { display: none !important; }
  .flt-tree .xp-rename { flex: none; margin-left: auto; background: none; border: none;
                         color: #64748b; cursor: pointer; font-size: 12px; padding: 0 2px; }
  .flt-tree .xp-rename:hover { color: #e2e8f0; }
  .flt-tree .xp-alias { color: #93c5fd; }
  td.mname { cursor: text; }
  tr.ds-empty, .ds-hidden { display: none !important; }

  /* Main content */
  main { margin-left: 360px; padding: 28px 32px; flex: 1; max-width: calc(100vw - 360px); }

  /* Sections */
  section.category { margin-bottom: 36px; }
  section.category > h2 { font-size: 20px; color: #1e293b; border-bottom: 2px solid #3b82f6;
                           padding-bottom: 6px; margin-bottom: 16px; }
  details { margin-bottom: 20px; }
  details > summary { cursor: pointer; font-size: 15px; font-weight: 600; color: #475569;
                       padding: 6px 0; user-select: none; }
  details > summary:hover { color: #1e40af; }
  .figure-wrapper { margin: 12px 0; overflow-x: auto; }

  /* Custom aligned cell tooltip (replaces the native title="" tooltip so that
     sub-task scores line up in columns regardless of task-name length). */
  #celltip { position: fixed; z-index: 9999; pointer-events: none; display: none;
             background: #0f172a; color: #e2e8f0; border: 1px solid #334155;
             border-radius: 6px; padding: 8px 10px; box-shadow: 0 6px 18px rgba(0,0,0,.35);
             font-size: 12px; line-height: 1.5; max-width: 460px; }
  #celltip.show { display: block; }
  #celltip .ct-head { font-weight: 700; color: #f8fafc; white-space: nowrap; }
  #celltip .ct-head + .ct-head { font-weight: 400; color: #cbd5e1; }
  #celltip .ct-grid { display: grid; grid-template-columns: auto max-content max-content;
                      column-gap: 12px; row-gap: 1px; margin-top: 3px; }
  #celltip .ct-k { color: #94a3b8; white-space: nowrap; }
  #celltip .ct-v { text-align: right; white-space: nowrap;
                   font-variant-numeric: tabular-nums; }
  #celltip .ct-r { color: #64748b; white-space: nowrap; }
</style>
</head>
<body>
<nav class="sidebar">
  <h2>AudioBench Results</h2>
  <ul>
__NAV_ITEMS__
  </ul>
  <div id="xp-filter" class="flt-panel">
    <div class="xp-filter-head">
      <span>Experiments</span>
      <div class="flt-btns">
        <button id="xp-rename-reset" type="button" title="Annuler tous les renommages">Noms d'origine</button>
        <button id="xp-toggle-all" type="button">Tout décocher</button>
      </div>
    </div>
    <div id="xp-filter-tree" class="flt-tree"></div>
  </div>
  <div id="ds-filter" class="flt-panel">
    <div class="xp-filter-head">
      <span>Datasets</span>
      <div class="flt-btns">
        <button id="ds-reset" type="button" title="Sélection par défaut">Défaut</button>
        <button id="ds-core" type="button" title="Sélection par défaut, limitée aux tâches ASR, AST et QA">ASR/AST/QA</button>
        <button id="ds-toggle-all" type="button">Tout cocher</button>
      </div>
    </div>
    <div id="ds-filter-tree" class="flt-tree"></div>
  </div>
</nav>
<main>
__SECTIONS__
</main>
<div id="celltip"></div>
<script type="application/json" id="report-data">__REPORT_DATA__</script>
<script>
(function () {
  var tip = document.getElementById('celltip');
  var NL = String.fromCharCode(10);

  function stripEnd(str) {
    while (str.length && str.charAt(str.length - 1) === ' ') str = str.slice(0, -1);
    return str;
  }

  function render(text) {
    if (window.xpRenameText) text = window.xpRenameText(text);
    tip.textContent = '';
    var grid = null;
    text.split(NL).forEach(function (line) {
      if (line === '') return;
      var idx = line.indexOf(': ');
      if (idx === -1) {                       // header / summary line -> full width
        grid = null;
        var h = document.createElement('div');
        h.className = 'ct-head';
        h.textContent = line;
        tip.appendChild(h);
        return;
      }
      if (!grid) {                            // start a fresh aligned block
        grid = document.createElement('div');
        grid.className = 'ct-grid';
        tip.appendChild(grid);
      }
      var k = line.slice(0, idx), v = line.slice(idx + 2), r = '';
      if (v.charAt(v.length - 1) === ')') {   // peel a trailing "(3e)" rank marker
        var op = v.lastIndexOf('(');
        if (op !== -1) { r = v.slice(op + 1, v.length - 1); v = stripEnd(v.slice(0, op)); }
      }
      var ke = document.createElement('span'); ke.className = 'ct-k'; ke.textContent = k;
      var ve = document.createElement('span'); ve.className = 'ct-v'; ve.textContent = v;
      var re = document.createElement('span'); re.className = 'ct-r'; re.textContent = r;
      grid.appendChild(ke); grid.appendChild(ve); grid.appendChild(re);
    });
  }

  function position(e) {
    var pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    var x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > window.innerWidth - 8)  x = e.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = e.clientY - h - pad;
    tip.style.left = Math.max(4, x) + 'px';
    tip.style.top  = Math.max(4, y) + 'px';
  }

  // Move native title="" onto data-tip so the browser tooltip does not compete.
  document.querySelectorAll('td[title], th[title]').forEach(function (el) {
    el.setAttribute('data-tip', el.getAttribute('title'));
    el.removeAttribute('title');
  });

  document.addEventListener('mouseover', function (e) {
    var el = e.target.closest('[data-tip]');
    if (!el) return;
    render(el.getAttribute('data-tip'));
    tip.classList.add('show');
    position(e);
  });
  document.addEventListener('mousemove', function (e) {
    if (tip.classList.contains('show')) position(e);
  });
  document.addEventListener('mouseout', function (e) {
    var el = e.target.closest('[data-tip]');
    if (!el) return;
    if (e.relatedTarget && el.contains(e.relatedTarget)) return;
    tip.classList.remove('show');
  });
})();
</script>
<script>
(function () {
  var rows = Array.prototype.slice.call(document.querySelectorAll('tr[data-model]'));
  var models = [];
  rows.forEach(function (tr) {
    var m = tr.getAttribute('data-model');
    if (models.indexOf(m) === -1) models.push(m);
  });
  models.sort();
  if (!models.length) return;

  var treeEl = document.getElementById('xp-filter-tree');
  var toggleBtn = document.getElementById('xp-toggle-all');
  var checked = {};
  models.forEach(function (m) { checked[m] = true; });

  // -------------------------------------------------------------------
  // Build a trie of models, tokenized on "/" and "_" (keeping the
  // delimiter that preceded each token), so experiments that share a
  // path/name prefix ("LINAGORA/Canary_Luciole-1B_..._v3_buckets_...")
  // group and nest automatically -- no hardcoded naming knowledge needed.
  // -------------------------------------------------------------------
  function tokenize(name) {
    var parts = name.split(/([/_])/);
    var tokens = [], delims = [];
    for (var i = 0; i < parts.length; i += 2) tokens.push(parts[i]);
    for (var j = 1; j < parts.length; j += 2) delims.push(parts[j]);
    return { tokens: tokens, delims: delims };
  }

  function newNode() { return { children: new Map(), models: [] }; }
  var root = newNode();
  models.forEach(function (m) {
    var t = tokenize(m);
    var cur = root;
    for (var i = 0; i < t.tokens.length; i++) {
      var tok = t.tokens[i];
      var delim = i === 0 ? null : t.delims[i - 1];
      if (!cur.children.has(tok)) cur.children.set(tok, { delim: delim, node: newNode() });
      cur = cur.children.get(tok).node;
    }
    cur.models.push(m);
  });

  // Compress chains of single, model-less children into one label so the
  // tree only branches where models actually diverge.
  function compressChildren(node) {
    var out = [];
    node.children.forEach(function (edge, tok) {
      var label = (edge.delim || '') + tok;
      var child = edge.node;
      while (child.models.length === 0 && child.children.size === 1) {
        var onlyTok, onlyEdge;
        child.children.forEach(function (e, t) { onlyTok = t; onlyEdge = e; });
        label += (onlyEdge.delim || '') + onlyTok;
        child = onlyEdge.node;
      }
      out.push({ label: label, node: child });
    });
    return out;
  }

  // -------------------------------------------------------------------
  // Render. Returns the list of leaf model names under the rendered node,
  // and registers group checkboxes so their tri-state can be refreshed.
  // -------------------------------------------------------------------
  var groupBoxes = []; // { cb, leaves: [model,...] }
  var leafBoxes = {};  // model -> checkbox element
  var leafAliases = {}; // model -> span showing its display name when renamed

  // -------------------------------------------------------------------
  // Display-only renaming (for clean screenshots, e.g. model cards).
  // data-model and the filter tree keep the real name; only the visible
  // name cells, tooltips and Plotly labels change. Saved per browser.
  // -------------------------------------------------------------------
  var RENAME_KEY = 'audiobench-xp-renames';
  var renames = {};
  try { renames = JSON.parse(localStorage.getItem(RENAME_KEY) || '{}') || {}; } catch (e) { renames = {}; }

  function displayName(m) { return renames[m] || m; }

  function saveRenames() {
    try { localStorage.setItem(RENAME_KEY, JSON.stringify(renames)); } catch (e) {}
  }

  function askRename(m) {
    var v = window.prompt('Nouveau nom pour ' + m + ' (vide = nom d\\'origine) :', displayName(m));
    if (v === null) return;
    v = v.trim();
    if (v && v !== m) renames[m] = v; else delete renames[m];
    saveRenames();
    applyRenames();
  }

  // Longest names first so a name that prefixes another is not replaced inside it.
  window.xpRenameText = function (text) {
    Object.keys(renames).sort(function (a, b) { return b.length - a.length; }).forEach(function (m) {
      text = text.split(m).join(renames[m]);
    });
    return text;
  };

  var isModel = {};
  models.forEach(function (m) { isModel[m] = true; });

  function hasModel(v) {
    if (typeof v === 'string') return isModel[v] === true;
    return Array.isArray(v) && v.some(function (x) { return typeof x === 'string' && isModel[x] === true; });
  }

  function renameValue(v) {
    if (typeof v === 'string') return isModel[v] === true ? displayName(v) : v;
    if (Array.isArray(v)) return v.map(renameValue);
    return v;
  }

  // Each figure remembers, once, the original value of every trace field that
  // holds a model name; renaming rewrites those fields in place and redraws
  // only the figures that reference a model (one redraw per figure).
  var PLOT_FIELDS = ['name', 'x', 'y', 'text', 'hovertext', 'legendgroup', 'labels', 'theta'];
  function renamePlots() {
    if (!window.Plotly) return;
    document.querySelectorAll('.js-plotly-plot').forEach(function (gd) {
      if (!gd.data) return;
      if (!gd._xpOrig) {
        if (!Object.keys(renames).length) return;
        gd._xpOrig = [];
        gd.data.forEach(function (tr, i) {
          PLOT_FIELDS.forEach(function (f) {
            if (hasModel(tr[f])) gd._xpOrig.push({ i: i, f: f, v: tr[f] });
          });
        });
      }
      if (!gd._xpOrig.length) return;
      gd._xpOrig.forEach(function (o) { gd.data[o.i][o.f] = renameValue(o.v); });
      window.Plotly.redraw(gd);
    });
  }

  function applyRenames() {
    rows.forEach(function (tr) {
      var td = tr.querySelector('td.mname');
      if (td) td.textContent = displayName(tr.getAttribute('data-model'));
    });
    models.forEach(function (m) {
      var a = leafAliases[m];
      if (a) a.textContent = renames[m] ? ' → ' + renames[m] : '';
    });
    renamePlots();
  }

  document.addEventListener('dblclick', function (e) {
    var td = e.target.closest('td.mname');
    if (!td || !td.parentNode.hasAttribute('data-model')) return;
    askRename(td.parentNode.getAttribute('data-model'));
  });

  document.getElementById('xp-rename-reset').addEventListener('click', function () {
    if (!Object.keys(renames).length) return;
    if (!window.confirm('Revenir aux noms d\\'origine pour toutes les expériences ?')) return;
    renames = {};
    saveRenames();
    applyRenames();
  });

  function renderLeaf(container, modelName) {
    var li = document.createElement('li');
    var label = document.createElement('label');
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.addEventListener('change', function () {
      checked[modelName] = cb.checked;
      refreshGroups();
      applyFilter();
    });
    leafBoxes[modelName] = cb;
    label.appendChild(cb);
    var name = document.createElement('span');
    name.textContent = modelName;
    label.appendChild(name);
    var alias = document.createElement('span');
    alias.className = 'xp-alias';
    label.appendChild(alias);
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'xp-rename';
    btn.title = 'Renommer (affichage uniquement)';
    btn.textContent = '✎';
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      askRename(modelName);
    });
    label.appendChild(btn);
    leafAliases[modelName] = alias;
    li.appendChild(label);
    container.appendChild(li);
  }

  function renderNode(container, label, node, depth) {
    var childEntries = compressChildren(node);
    var isPureLeaf = node.models.length > 0 && childEntries.length === 0;

    if (isPureLeaf) {
      node.models.forEach(function (m) { renderLeaf(container, m); });
      return node.models.slice();
    }

    var li = document.createElement('li');
    var details = document.createElement('details');
    details.open = depth < 1;
    var summary = document.createElement('summary');
    var caret = document.createElement('span');
    caret.className = 'xp-caret';
    caret.textContent = '▸';
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    var text = document.createElement('span');
    text.className = 'xp-group';
    text.textContent = label;
    summary.appendChild(caret);
    summary.appendChild(cb);
    summary.appendChild(text);
    var count = document.createElement('span');
    count.className = 'xp-count';
    summary.appendChild(count);
    details.appendChild(summary);

    var ul = document.createElement('ul');
    var leaves = [];
    // A model that terminates exactly at this group node (rare: its name
    // is itself a prefix of a sibling's name) is shown as its own leaf too.
    node.models.forEach(function (m) {
      renderLeaf(ul, m);
      leaves.push(m);
    });
    childEntries.forEach(function (entry) {
      leaves = leaves.concat(renderNode(ul, entry.label, entry.node, depth + 1));
    });
    details.appendChild(ul);
    li.appendChild(details);
    container.appendChild(li);

    count.textContent = '(' + leaves.length + ')';
    cb.addEventListener('click', function (e) { e.stopPropagation(); });
    cb.addEventListener('change', function () {
      leaves.forEach(function (m) {
        checked[m] = cb.checked;
        var lb = leafBoxes[m];
        if (lb) lb.checked = cb.checked;
      });
      refreshGroups();
      applyFilter();
    });
    groupBoxes.push({ cb: cb, leaves: leaves });
    return leaves;
  }

  var rootUl = document.createElement('ul');
  compressChildren(root).forEach(function (entry) {
    renderNode(rootUl, entry.label, entry.node, 0);
  });
  treeEl.appendChild(rootUl);

  function refreshGroups() {
    groupBoxes.forEach(function (g) {
      var n = g.leaves.filter(function (m) { return checked[m]; }).length;
      g.cb.checked = n === g.leaves.length;
      g.cb.indeterminate = n > 0 && n < g.leaves.length;
    });
  }

  function applyFilter() {
    rows.forEach(function (tr) {
      var m = tr.getAttribute('data-model');
      tr.classList.toggle('xp-hidden', !checked[m]);
    });
    if (window.refreshReportTables) window.refreshReportTables();
    var allChecked = models.every(function (m) { return checked[m]; });
    toggleBtn.textContent = allChecked ? 'Tout décocher' : 'Tout cocher';
  }

  toggleBtn.addEventListener('click', function () {
    var allChecked = models.every(function (m) { return checked[m]; });
    var next = !allChecked;
    models.forEach(function (m) { checked[m] = next; });
    Object.keys(leafBoxes).forEach(function (m) { leafBoxes[m].checked = next; });
    refreshGroups();
    applyFilter();
  });

  refreshGroups();
  applyRenames();
})();
</script>
<script>
(function () {
  // ===================================================================
  // Table engine: every score cell carries data-f = the id of the node
  // (in the score graph emitted by the Python side) it displays. Leaves are
  // (model, dataset) scores; means and display scaling are inner nodes. When
  // datasets or models are toggled, every cell, CI, rank colour, tooltip,
  // aggregate column and row order is recomputed from that graph.
  // ===================================================================
  var DATA = JSON.parse(document.getElementById('report-data').textContent);
  var NODES = DATA.nodes;
  var COLORS = __AGG_COLORS__;
  var NL = String.fromCharCode(10);
  var dsOn = DATA.datasets.map(function () { return true; });
  DATA.off.forEach(function (i) { dsOn[i] = false; });

  // --- Graph evaluation (memoized per refresh) ---
  var memo = new Array(NODES.length);
  var ascMemo = new Array(NODES.length);

  // sum() as CPython >= 3.12 computes it for floats (Neumaier compensation),
  // so means tie exactly where they tie on the Python side.
  function pysum(xs) {
    var s = 0, c = 0;
    for (var i = 0; i < xs.length; i++) {
      var x = xs[i], t = s + x;
      if (Math.abs(s) >= Math.abs(x)) c += (s - t) + x; else c += (x - t) + s;
      s = t;
    }
    return c && isFinite(c) ? s + c : s;
  }

  function value(id) {
    if (memo[id] !== undefined) return memo[id];
    var n = NODES[id], v = null;
    if (n[0] === 'L') {
      v = dsOn[n[1]] ? n[2] : null;
    } else if (n[0] === 'M') {
      var xs = [];
      n[1].forEach(function (c) {
        var cv = value(c);
        if (cv !== null) xs.push(cv);
      });
      v = xs.length ? pysum(xs) / xs.length : null;
    } else {  // 'D'
      var x = value(n[1]);
      v = x === null ? null : Math.min(n[2] ? x * 100 : x, 100);
    }
    memo[id] = v;
    return v;
  }

  function isAsc(id) {  // lower is better (mirrors task_ascending)
    if (ascMemo[id] !== undefined) return ascMemo[id];
    var n = NODES[id], a;
    if (n[0] === 'L') a = n[3];
    else if (n[0] === 'D') a = isAsc(n[1]);
    else a = n[1].every(isAsc);
    ascMemo[id] = a;
    return a;
  }

  // CI half-width in display units, or null. A leaf uses its own std; a mean
  // pools the per-sample scores of its enabled leaves (like np.std(pooled)).
  function ciOf(id, pct) {
    var n = NODES[id];
    if (n[0] === 'D') n = NODES[id = n[1]];
    var std, cnt;
    if (n[0] === 'L') {
      if (!dsOn[n[1]] || n[4][3] === null || !n[4][0]) return null;
      std = n[4][3]; cnt = n[4][0];
    } else {
      var acc = { n: 0, s: 0, ss: 0 };
      (function gather(i) {
        var x = NODES[i];
        if (x[0] === 'L') {
          if (dsOn[x[1]] && x[4][0]) { acc.n += x[4][0]; acc.s += x[4][1]; acc.ss += x[4][2]; }
        } else if (x[0] === 'D') gather(x[1]);
        else x[1].forEach(gather);
      })(id);
      if (!acc.n) return null;
      var mean = acc.s / acc.n;
      std = Math.sqrt(Math.max(0, acc.ss / acc.n - mean * mean));
      cnt = acc.n;
    }
    var ci = 1.96 * std / Math.sqrt(cnt);
    return { ci: pct ? ci * 100 : ci, n: cnt };
  }

  // Match Python's f"{v:.nf}" (round-half-even on exact ties).
  var formatters = {};
  function fmt(v, digits) {
    var f = formatters[digits];
    if (f === undefined) {
      try {
        f = new Intl.NumberFormat('en-US', {
          minimumFractionDigits: digits, maximumFractionDigits: digits,
          roundingMode: 'halfEven', useGrouping: false });
      } catch (e) { f = null; }
      formatters[digits] = f;
    }
    return f ? f.format(v) : v.toFixed(digits);
  }

  function aggVal(x) {  // aggregate payload value: inline float or {n: node id}
    return typeof x === 'number' ? x : value(x.n);
  }

  // --- Tables ---
  var tables = Array.prototype.slice.call(document.querySelectorAll('table.ov-tbl'))
    .filter(function (t) { return t.tBodies.length && t.querySelector('td[data-f]'); })
    .map(function (tbl) {
      var payload = document.querySelector('script.agg-data[data-table="' + tbl.id + '"]');
      // Column index -> header cell (data columns span both header rows).
      var headers = {};
      if (tbl.tHead && tbl.tHead.rows.length) {
        var c = 0;
        Array.prototype.forEach.call(tbl.tHead.rows[0].cells, function (th) {
          if (th.rowSpan === 2) headers[c] = th;
          c += th.colSpan;
        });
      }
      var cols = {};
      Array.prototype.forEach.call(tbl.querySelectorAll('td[data-f]'), function (td) {
        (cols[td.cellIndex] = cols[td.cellIndex] || []).push(td);
      });
      return { tbl: tbl, headers: headers, cols: cols,
               agg: payload ? JSON.parse(payload.textContent) : null };
    });

  function rowOn(tr) { return !tr.classList.contains('xp-hidden'); }

  function computeAggs(data, on) {
    var rank = {}, mm = {}, zs = {};
    data.items.forEach(function (item) {
      var ms = [], hib = [], raw = {};
      Object.keys(item.s).forEach(function (m) {
        if (!on[m]) return;
        var v = aggVal(item.s[m]);
        if (v === null) return;
        ms.push(m); hib.push(item.asc ? 100 - v : v);
        raw[m] = item.r ? aggVal(item.r[m]) : v;
      });
      if (!ms.length) return;
      var lo = Math.min.apply(null, hib), hi = Math.max.apply(null, hib);
      var mean = hib.reduce(function (a, b) { return a + b; }, 0) / hib.length;
      var std = Math.sqrt(hib.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0) / hib.length);
      // Avg Rank uses the unclamped scores when provided ("r").
      ms.slice().sort(function (a, b) { return item.asc ? raw[a] - raw[b] : raw[b] - raw[a]; })
        .forEach(function (m, r) { (rank[m] = rank[m] || []).push(r + 1); });
      ms.forEach(function (m, i) {
        var v = hib[i];
        (mm[m] = mm[m] || []).push(hi > lo ? (v - lo) / (hi - lo) : 1);
        (zs[m] = zs[m] || []).push(std > 0 ? (v - mean) / std : 0);
      });
    });
    function avg(obj) {
      var out = {};
      Object.keys(obj).forEach(function (m) {
        out[m] = pysum(obj[m]) / obj[m].length;
      });
      return out;
    }
    return { avg_rank: avg(rank), minmax: avg(mm), zscore: avg(zs) };
  }

  // Colour 1st / 2nd / before-last / last among *ranked* rows (best first).
  function rankColors(ranked) {
    var color = new Map(), n = ranked.length;
    if (n >= 1) color.set(ranked[0], COLORS.first);
    if (n >= 2) color.set(ranked[1], COLORS.second);
    if (n >= 3) color.set(ranked[n - 1], COLORS.last);
    if (n >= 4) color.set(ranked[n - 2], COLORS.before_last);
    return color;
  }

  // Only touch the DOM when a cell actually changes (keeps toggling fast).
  function setCell(td, html, bg) {
    if (td._html !== html) { td.innerHTML = html; td._html = html; }
    if (td._bg !== bg) { td.style.background = bg; td._bg = bg; }
  }

  function setMissing(td) {
    setCell(td, '-', COLORS.missing);
    td.removeAttribute('data-tip');
    td._tip = undefined;
  }

  function fillTooltip(template, ranks) {
    var out = [];
    template.split(NL).forEach(function (line) {
      var dropped = false;
      var text = line.replace(/\\[\\[([^\\]]*)\\]\\]/g, function (_, tok) {
        var p = tok.split(':'), id = +p[1];
        if (p[0] === 'V') {
          var v = value(id);
          if (v === null) { dropped = true; return ''; }
          return fmt(v, 2);
        }
        if (p[0] === 'R') {
          var r = ranks[id];
          if (!r) return '';
          return p[2] === 'p' ? r + 'e — ' : ' (' + r + 'e)';
        }
        if (p[0] === 'C') {
          var ci = value(id) === null ? null : ciOf(id, p[2] === '1');
          if (!ci) return '';
          if (p[3] === 's') return '±' + fmt(ci.ci, 2);
          var v0 = value(id);
          return ' [' + fmt(v0 - ci.ci, 2) + ', ' + fmt(v0 + ci.ci, 2) + '], n=' + ci.n;
        }
        return '';
      });
      if (!dropped) out.push(text);
    });
    return out.join(NL);
  }

  function refreshTable(t) {
    var body = t.tbl.tBodies[0];
    var trs = Array.prototype.slice.call(body.rows).filter(function (tr) { return tr.hasAttribute('data-model'); });
    var on = {};
    trs.forEach(function (tr) { if (rowOn(tr)) on[tr.getAttribute('data-model')] = true; });

    // 1. Aggregates + row order (by the first aggregate).
    if (t.agg) {
      var vals = computeAggs(t.agg, on);
      t.agg.aggs.forEach(function (agg) {
        var v = vals[agg.name];
        var ranked = trs.filter(function (tr) { return on[tr.getAttribute('data-model')] && tr.getAttribute('data-model') in v; })
          .sort(function (a, b) {
            var d = v[a.getAttribute('data-model')] - v[b.getAttribute('data-model')];
            return agg.hib ? -d : d;
          });
        var color = rankColors(ranked);
        trs.forEach(function (tr) {
          var td = tr.querySelector('td[data-agg="' + agg.name + '"]');
          if (!td) return;
          var m = tr.getAttribute('data-model');
          if (m in v) setCell(td, fmt(v[m], agg.digits), color.get(tr) || '');
          else setCell(td, '-', COLORS.missing);
        });
      });
      var first = t.agg.aggs[0];
      if (first) {
        var fv = vals[first.name];
        trs.sort(function (a, b) {
          var ma = a.getAttribute('data-model'), mb = b.getAttribute('data-model');
          var ha = ma in fv, hb = mb in fv;
          if (ha !== hb) return ha ? -1 : 1;
          if (!ha) return 0;
          var d = fv[ma] - fv[mb];
          return first.hib ? -d : d;
        });
        var same = trs.every(function (tr, i) { return body.rows[i] === tr; });
        if (!same) trs.forEach(function (tr) { body.appendChild(tr); });
      }
    }

    // 2. Per-column values, ranks and colours over the visible rows.
    var ranks = {};       // node id -> rank within its column
    var rowHasData = new Map();
    Object.keys(t.cols).forEach(function (ci) {
      var cells = t.cols[ci];
      var asc = isAsc(+cells[0].getAttribute('data-f'));
      var present = cells.filter(function (td) {
        var ok = value(+td.getAttribute('data-f')) !== null;
        if (ok) rowHasData.set(td.parentNode, true);
        return ok && rowOn(td.parentNode);
      });
      // Stable sort in (current) row order, as Python's _full_ranks.
      present.sort(function (a, b) { return a.parentNode.rowIndex - b.parentNode.rowIndex; });
      var ranked = present.slice().sort(function (a, b) {
        var d = value(+a.getAttribute('data-f')) - value(+b.getAttribute('data-f'));
        return asc ? d : -d;
      });
      ranked.forEach(function (td, i) { ranks[+td.getAttribute('data-f')] = i + 1; });
      var color = rankColors(ranked);
      cells.forEach(function (td) {
        var id = +td.getAttribute('data-f');
        var v = value(id);
        if (v === null) { setMissing(td); return; }
        var html = fmt(v, 2);
        if (td.hasAttribute('data-ci')) {
          var c = ciOf(id, td.getAttribute('data-ci') === '1');
          if (c) html += ' <span class="ci">±' + fmt(c.ci, 2) + '</span>';
        }
        setCell(td, html, color.get(td) || '');
      });
      // Hide a column with no value for any visible model.
      var hide = !present.length;
      trs.forEach(function (tr) {  // includes the "-" cells of models without data
        if (tr.cells[ci]) tr.cells[ci].classList.toggle('ds-hidden', hide);
      });
      if (t.headers[ci]) t.headers[ci].classList.toggle('ds-hidden', hide);
    });

    // 3. Tooltips (need the ranks of every column).
    Object.keys(t.cols).forEach(function (ci) {
      t.cols[ci].forEach(function (td) {
        var tt = td.getAttribute('data-tt');
        if (!tt || value(+td.getAttribute('data-f')) === null) return;
        var tip = fillTooltip(tt, ranks);
        if (td._tip !== tip) { td.setAttribute('data-tip', tip); td._tip = tip; }
        td.removeAttribute('title');
      });
    });

    // 4. Rows without any value left (all their datasets unchecked).
    trs.forEach(function (tr) { tr.classList.toggle('ds-empty', !rowHasData.get(tr)); });
    var anyCol = Object.keys(t.cols).some(function (ci) {
      return !t.cols[ci][0].classList.contains('ds-hidden');
    });
    t.tbl.classList.toggle('ds-empty-tbl', !anyCol);
  }

  // Hide figure wrappers whose tables are all empty, then empty sections.
  function refreshSections() {
    document.querySelectorAll('.figure-wrapper').forEach(function (w) {
      var tbls = w.querySelectorAll('table.ov-tbl');
      var empty = tbls.length > 0 && Array.prototype.every.call(tbls, function (tb) {
        return tb.classList.contains('ds-empty-tbl');
      });
      w.classList.toggle('ds-hidden', empty);
    });
    document.querySelectorAll('section.category').forEach(function (sec) {
      var ws = sec.querySelectorAll('.figure-wrapper');
      var empty = ws.length > 0 && Array.prototype.every.call(ws, function (w) {
        return w.classList.contains('ds-hidden');
      });
      sec.classList.toggle('ds-hidden', empty);
      var link = document.querySelector('nav.sidebar a[href="#' + sec.id + '"]');
      if (link) link.parentNode.classList.toggle('ds-hidden', empty);
    });
  }

  function refresh() {
    memo = new Array(NODES.length);
    tables.forEach(refreshTable);
    refreshSections();
  }
  window.refreshReportTables = refresh;

  // ===================================================================
  // Dataset filter panel: Task -> "LANG · dataset" checkboxes.
  // ===================================================================
  var treeEl = document.getElementById('ds-filter-tree');
  var toggleBtn = document.getElementById('ds-toggle-all');
  var resetBtn = document.getElementById('ds-reset');
  var leafBoxes = [];   // dataset idx -> checkbox
  var groupBoxes = [];  // { cb, leaves: [idx] }

  // Task -> language -> dataset. Tasks with a single language skip the
  // language level (their leaves read "LANG · dataset").
  function langKey(l) {  // FR, EN first, like the tables
    var i = ['FR', 'EN'].indexOf(l.split('-')[0]);
    return (i === -1 ? '2' : '' + i) + l;
  }

  function groupNode(container, label, idxs) {  // returns the <ul> for children
    var li = document.createElement('li');
    var details = document.createElement('details');
    var summary = document.createElement('summary');
    var caret = document.createElement('span');
    caret.className = 'xp-caret';
    caret.textContent = '▸';
    var gcb = document.createElement('input');
    gcb.type = 'checkbox';
    var text = document.createElement('span');
    text.className = 'xp-group';
    text.textContent = label;
    var count = document.createElement('span');
    count.className = 'xp-count';
    summary.appendChild(caret); summary.appendChild(gcb);
    summary.appendChild(text); summary.appendChild(count);
    details.appendChild(summary);
    var ul = document.createElement('ul');
    details.appendChild(ul);
    li.appendChild(details);
    container.appendChild(li);
    gcb.addEventListener('click', function (e) { e.stopPropagation(); });
    gcb.addEventListener('change', function () {
      idxs.forEach(function (i) { dsOn[i] = gcb.checked; });
      update();
    });
    groupBoxes.push({ cb: gcb, leaves: idxs, count: count });
    return ul;
  }

  function leafNode(container, i, label) {
    var li = document.createElement('li');
    var lab = document.createElement('label');
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.addEventListener('change', function () { dsOn[i] = cb.checked; update(); });
    leafBoxes[i] = cb;
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(label));
    li.appendChild(lab);
    container.appendChild(li);
  }

  var byTask = {};
  DATA.datasets.forEach(function (d, i) {
    var langs = byTask[d[0]] = byTask[d[0]] || {};
    (langs[d[1]] = langs[d[1]] || []).push(i);
  });
  var rootUl = document.createElement('ul');
  Object.keys(byTask).sort().forEach(function (task) {
    var langs = Object.keys(byTask[task]).sort(function (a, b) {
      return langKey(a) < langKey(b) ? -1 : langKey(a) > langKey(b) ? 1 : 0;
    });
    var byName = function (a, b) {
      var x = DATA.datasets[a][2], y = DATA.datasets[b][2];
      return x < y ? -1 : x > y ? 1 : 0;
    };
    var all = [];
    langs.forEach(function (l) { all = all.concat(byTask[task][l].sort(byName)); });
    var taskUl = groupNode(rootUl, task, all);
    if (langs.length === 1) {
      all.forEach(function (i) {
        leafNode(taskUl, i, DATA.datasets[i][1] + ' · ' + DATA.datasets[i][2]);
      });
      return;
    }
    langs.forEach(function (l) {
      var idxs = byTask[task][l];
      var langUl = groupNode(taskUl, l, idxs);
      idxs.forEach(function (i) { leafNode(langUl, i, DATA.datasets[i][2]); });
    });
  });
  treeEl.appendChild(rootUl);

  function syncBoxes() {
    leafBoxes.forEach(function (cb, i) { if (cb) cb.checked = dsOn[i]; });
    groupBoxes.forEach(function (g) {
      var n = g.leaves.filter(function (i) { return dsOn[i]; }).length;
      g.cb.checked = n === g.leaves.length;
      g.cb.indeterminate = n > 0 && n < g.leaves.length;
      g.count.textContent = '(' + n + '/' + g.leaves.length + ')';
    });
    toggleBtn.textContent = dsOn.every(Boolean) ? 'Tout décocher' : 'Tout cocher';
  }

  function update() { syncBoxes(); refresh(); }

  toggleBtn.addEventListener('click', function () {
    var next = !dsOn.every(Boolean);
    dsOn = dsOn.map(function () { return next; });
    update();
  });
  function selectDefault(superCats) {
    dsOn = DATA.super_cats.map(function (sc) { return !superCats || superCats.indexOf(sc) !== -1; });
    DATA.off.forEach(function (i) { dsOn[i] = false; });
    update();
  }
  resetBtn.addEventListener('click', function () { selectDefault(null); });
  document.getElementById('ds-core').addEventListener('click', function () {
    selectDefault(['ASR', 'AST', 'QA']);
  });

  update();
})();
</script>
</body>
</html>
"""


def _slug(text):
    """Turn a category name into a URL-safe anchor id."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', text).strip('_')


def build_html_report(collected_figures, output_path, default_off_datasets=()):
    """Assemble a single HTML report from collected Plotly figures.

    Figures are grouped by category, with violin plots shown before tables
    within each group.  The sidebar is split into **Overview**, **Tasks**,
    and **Languages** groups.

    Symbolic table cells are resolved into data-* attributes and the score
    graph is embedded for the dataset filter; *default_off_datasets* lists the
    dataset indices unchecked when the page loads.
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

    html = _HTML_TEMPLATE.replace('__NAV_ITEMS__', '\n'.join(nav_lines))
    html = html.replace('__SECTIONS__', '\n'.join(section_blocks))
    report_data = json.dumps({
        "nodes": _SYM_NODES,
        "datasets": [list(k) for k in _SYM_DATASETS],
        "super_cats": [_super_category(k[0]) for k in _SYM_DATASETS],
        "off": list(default_off_datasets),
    }, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace('__REPORT_DATA__', report_data)
    html = html.replace('__AGG_COLORS__', json.dumps({**RANK_COLORS, "missing": MISSING_COLOR}))

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

    # --- Step 3: Language sections (French, English, Others) ---
    plot_language_sections(entries, collector, include_violin=args.violin,
                           table_aggregates=args.table_aggregates,
                           violin_entries=plot_entries)

    if not collector:
        print("No figures generated.")
        return

    output_path = os.path.join(args.output_folder, "report.html")
    build_html_report(collector, output_path, default_off_datasets=default_off)


if __name__ == "__main__":
    main()
