"""Deterministic scoring for "Temporal" subtasks.

Used when references are in the *pure* form produced by the SLU variant
generator (verified_test_pure.jsonl). Each subtask has a strict reference
shape that lets us bypass an LLM judge:

    word2time              "10.2s" | "10.2s,25.6s" | "10.2s-12.3s" | "not_found"
    word2time_first        "10.2s" | "not_found"
    time2word              "<word>" | "invalid_timestamp"
    time2sentence          "<sentence>" | "invalid_timestamp"
    word2sentence          "<sentence>" (multi sentences joined by '|') | "not_found"
    answer_with_time       "<answer>|<time>s"        (or "<answer>|<time1>s,<time2>s")
    answer_with_source     "<answer>|<time>s|<sentence>"
    answer_json            canonical JSON object

Per-sample score is in [0, 100]. Output shape mirrors the judge metrics so
leaderboard / _score.json consumers stay unchanged.
"""

import json
import re

import evaluate

from audio_bench.dataset_src.eval_methods.metrics import build_metric_stats


_TIMESTAMP_TOLERANCE_S = 0.5
_TIMESTAMP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE)
_TIMESTAMP_RE_LOOSE = re.compile(r"(\d+(?:\.\d+)?)")
_NEGATIVE_PHRASES = (
    "not_found", "not found", "no occurrence", "does not appear",
    "not present", "no such", "n/a", "none", "nothing",
    "invalid_timestamp", "invalid timestamp", "no sentence",
    "out of range", "beyond the audio",
)

_METEOR = None


def _meteor():
    global _METEOR
    if _METEOR is None:
        _METEOR = evaluate.load("meteor")
    return _METEOR


def _normalize(text):
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = s.strip("\"'`“”‘’ ")
    s = re.sub(r"\s+", " ", s)
    return s


def _is_negative_phrase(text):
    n = _normalize(text)
    if not n:
        return False
    if n in {"not_found", "invalid_timestamp", "none", "n/a"}:
        return True
    return any(p in n for p in _NEGATIVE_PHRASES)


def _parse_ref_timestamps(reference):
    """Return one of:
        ("points", [floats])           — exact-match references
        ("range",  (lo, hi))            — interval reference
        ("negative", None)              — not_found
    """
    s = _normalize(reference)
    if _is_negative_phrase(s):
        return ("negative", None)
    # Range form first: "10.2s-12.3s"
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*-\s*(\d+(?:\.\d+)?)\s*s?\s*$", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if hi < lo:
            lo, hi = hi, lo
        return ("range", (lo, hi))
    pts = [float(x) for x in _TIMESTAMP_RE.findall(s)]
    if not pts:
        pts = [float(x) for x in _TIMESTAMP_RE_LOOSE.findall(s)]
    return ("points", pts)


def _parse_pred_timestamps(prediction):
    """Extract floats from prediction. Prefer s-suffixed; fall back to loose."""
    s = str(prediction or "")
    pts = [float(x) for x in _TIMESTAMP_RE.findall(s)]
    if pts:
        return pts
    return [float(x) for x in _TIMESTAMP_RE_LOOSE.findall(s)]


def _f1_with_tolerance(refs, preds, tol=_TIMESTAMP_TOLERANCE_S):
    if not refs and not preds:
        return 1.0
    if not refs or not preds:
        return 0.0
    used = [False] * len(preds)
    matched = 0
    for r in refs:
        best_i, best_d = -1, tol
        for i, p in enumerate(preds):
            if used[i]:
                continue
            d = abs(p - r)
            if d <= best_d:
                best_d = d
                best_i = i
        if best_i >= 0:
            used[best_i] = True
            matched += 1
    if matched == 0:
        return 0.0
    precision = matched / len(preds)
    recall = matched / len(refs)
    return 2 * precision * recall / (precision + recall)


def _score_word2time(reference, prediction, first_only=False):
    kind, payload = _parse_ref_timestamps(reference)
    if kind == "negative":
        return 100.0 if _is_negative_phrase(prediction) else 0.0
    pred_ts = _parse_pred_timestamps(prediction)
    if first_only and pred_ts:
        pred_ts = pred_ts[:1]
    if kind == "range":
        lo, hi = payload
        if not pred_ts:
            return 0.0
        inside = [t for t in pred_ts if lo - _TIMESTAMP_TOLERANCE_S <= t <= hi + _TIMESTAMP_TOLERANCE_S]
        if not inside:
            return 0.0
        endpoint_hits = sum(
            1 for ep in (lo, hi)
            if any(abs(t - ep) <= _TIMESTAMP_TOLERANCE_S for t in pred_ts)
        )
        if endpoint_hits == 2:
            return 100.0
        if endpoint_hits == 1:
            return 75.0
        return 50.0
    return 100.0 * _f1_with_tolerance(payload, pred_ts)


def _score_time2word(reference, prediction):
    if _is_negative_phrase(reference):
        return 100.0 if _is_negative_phrase(prediction) else 0.0
    ref_n = _normalize(reference)
    pred_n = _normalize(prediction)
    if not ref_n:
        return 0.0
    if pred_n == ref_n:
        return 100.0
    pred_tokens = pred_n.split()
    if len(pred_tokens) <= 5 and re.search(rf"(?<!\w){re.escape(ref_n)}(?!\w)", pred_n):
        return 50.0
    return 0.0


def _meteor_score(reference, prediction):
    """Return METEOR in [0, 100]."""
    ref = (reference or "").strip()
    pred = (prediction or "").strip()
    if not ref:
        return 0.0
    if not pred:
        return 0.0
    score = _meteor().compute(predictions=[pred], references=[ref])["meteor"]
    return 100.0 * float(score)


def _score_sentence(reference, prediction, multi=False):
    if _is_negative_phrase(reference):
        return 100.0 if _is_negative_phrase(prediction) else 0.0
    refs = [r.strip() for r in reference.split("|")] if multi else [reference]
    refs = [r for r in refs if r]
    if not refs:
        return 0.0
    scores = [_meteor_score(r, prediction) for r in refs]
    return sum(scores) / len(scores)


def _split_pipe(text, n):
    """Split text on '|' into exactly n parts, padding with empty strings."""
    parts = [p.strip() for p in str(text or "").split("|")]
    if len(parts) < n:
        parts += [""] * (n - len(parts))
    elif len(parts) > n:
        parts = parts[:n - 1] + ["|".join(parts[n - 1:])]
    return parts


def _score_answer_with_time(reference, prediction):
    if _is_negative_phrase(reference):
        return 100.0 if _is_negative_phrase(prediction) else 0.0
    ref_ans, ref_time = _split_pipe(reference, 2)
    pred_ans, pred_time = _split_pipe(prediction, 2)
    s_ans = _score_time2word(ref_ans, pred_ans)
    s_time = _score_word2time(ref_time, pred_time)
    return 0.5 * s_ans + 0.5 * s_time


def _score_answer_with_source(reference, prediction):
    if _is_negative_phrase(reference):
        return 100.0 if _is_negative_phrase(prediction) else 0.0
    ref_ans, ref_time, ref_sent = _split_pipe(reference, 3)
    pred_ans, pred_time, pred_sent = _split_pipe(prediction, 3)
    s_ans = _score_time2word(ref_ans, pred_ans)
    s_time = _score_word2time(ref_time, pred_time)
    s_sent = _score_sentence(ref_sent, pred_sent, multi=False)
    return 0.5 * s_ans + 0.25 * s_time + 0.25 * s_sent


def _score_json_field(ref_val, pred_val):
    """Score a single canonical JSON field. Heuristic: type-aware comparison."""
    if isinstance(ref_val, list):
        if not isinstance(pred_val, list):
            pred_val = [pred_val]
        ref_strs = [str(v) for v in ref_val]
        pred_strs = [str(v) for v in pred_val]
        ref_ts = []
        for v in ref_strs:
            ref_ts += _parse_pred_timestamps(v)
        if ref_ts:
            pred_ts = []
            for v in pred_strs:
                pred_ts += _parse_pred_timestamps(v)
            return 100.0 * _f1_with_tolerance(ref_ts, pred_ts)
        joined_ref = " | ".join(ref_strs)
        joined_pred = " | ".join(pred_strs)
        return _meteor_score(joined_ref, joined_pred)
    ref_s = str(ref_val) if ref_val is not None else ""
    pred_s = str(pred_val) if pred_val is not None else ""
    if _TIMESTAMP_RE.search(ref_s) or re.fullmatch(r"\s*\d+(?:\.\d+)?\s*s?\s*", ref_s):
        return _score_word2time(ref_s, pred_s)
    if len(ref_s.split()) <= 4:
        return _score_time2word(ref_s, pred_s)
    return _meteor_score(ref_s, pred_s)


def _score_answer_json(reference, prediction):
    if _is_negative_phrase(reference):
        return 100.0 if _is_negative_phrase(prediction) else 0.0
    try:
        ref_obj = json.loads(reference)
    except (TypeError, ValueError):
        return 0.0
    pred_obj = _try_parse_json(prediction)
    if pred_obj is None:
        return 0.0
    if not isinstance(ref_obj, dict) or not isinstance(pred_obj, dict):
        if ref_obj == pred_obj:
            return 100.0
        return _meteor_score(
            json.dumps(ref_obj, sort_keys=True), json.dumps(pred_obj, sort_keys=True)
        )
    if not ref_obj:
        return 100.0 if pred_obj == ref_obj else 0.0
    scores = []
    for k, v in ref_obj.items():
        if k in pred_obj:
            scores.append(_score_json_field(v, pred_obj[k]))
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def _try_parse_json(text):
    if text is None:
        return None
    s = str(text).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (TypeError, ValueError):
            return None
    return None


_SCORERS = {
    "word2time": lambda r, p: _score_word2time(r, p, first_only=False),
    "word2time_first": lambda r, p: _score_word2time(r, p, first_only=True),
    "time2word": _score_time2word,
    "time2sentence": lambda r, p: _score_sentence(r, p, multi=False),
    "word2sentence": lambda r, p: _score_sentence(r, p, multi=True),
    "answer_with_time": _score_answer_with_time,
    "answer_with_source": _score_answer_with_source,
    "answer_json": _score_answer_json,
}


def _score_one(sub_task, reference, prediction):
    key = (sub_task or "").strip().lower()
    scorer = _SCORERS.get(key)
    if scorer is None:
        return _score_time2word(reference, prediction), f"no scorer for sub_task={sub_task!r}, fell back to exact-match"
    return scorer(reference, prediction), ""


def compute_temporal_regex(references, predictions, sub_tasks):
    """Score Temporal predictions deterministically.

    Returns the same shape as judge metrics: {'temporal_regex': stats, 'details': [...]}
    """
    per_sample = []
    details = []
    for ref, pred, st in zip(references, predictions, sub_tasks):
        score, note = _score_one(st, ref, pred)
        per_sample.append(score)
        d = {
            "sub_task": st,
            "reference": ref,
            "prediction": pred,
            "rate_score": score,
        }
        if note:
            d["note"] = note
        details.append(d)
    stats = build_metric_stats(per_sample)
    stats["success_rate"] = 1.0
    return {"temporal_regex": stats, "details": details}
