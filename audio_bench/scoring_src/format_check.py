"""Deterministic format-following check via json.loads() and structural comparison."""

import json
import re


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(text):
    if not isinstance(text, str):
        return text
    text = _FENCE_RE.sub("", text.strip())
    return text.strip()


def _try_parse(text):
    text = _strip_fences(text)
    try:
        return json.loads(text), None
    except (json.JSONDecodeError, TypeError) as e:
        # Fall back to the first {...} or [...] block in the string.
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1)), None
            except json.JSONDecodeError as e2:
                return None, str(e2)
        return None, str(e)


def _shape_match(pred, ref):
    """Recursively check that pred has the same structural shape as ref.

    Dicts must share the exact key set; values are compared recursively.
    Lists are accepted as long as both sides are lists (length and item shape
    are not enforced — references in this dataset are single-instance examples).
    Leaves (str/int/float/bool/None) match by Python type, with int/float treated
    as interchangeable numeric leaves.
    """
    if isinstance(ref, dict):
        if not isinstance(pred, dict):
            return False
        if set(pred.keys()) != set(ref.keys()):
            return False
        return all(_shape_match(pred[k], ref[k]) for k in ref)
    if isinstance(ref, list):
        return isinstance(pred, list)
    if isinstance(ref, bool):
        return isinstance(pred, bool)
    if isinstance(ref, (int, float)):
        return isinstance(pred, (int, float)) and not isinstance(pred, bool)
    if ref is None:
        return pred is None
    return isinstance(pred, str)


def format_check_json(_model_path, input_data, task_type=None):
    """Score predictions on JSON-format compliance against the reference shape.

    Returns (results_dict, all_details) matching the signature used by the
    judge eval methods so it can be wrapped by BaseDatasetProcessor._enrich_judge.

    Per-sample score is 1 if the prediction parses as JSON AND its structure
    matches the reference's structure, else 0.
    """
    del task_type  # unused — format checking is structural, not semantic
    questions, references, predictions = input_data

    all_details = []
    for q, r, p in zip(questions, references, predictions):
        pred_parsed, pred_err = _try_parse(p)
        ref_parsed, _ = _try_parse(r)

        parses = pred_parsed is not None
        shape_ok = parses and ref_parsed is not None and _shape_match(pred_parsed, ref_parsed)
        score = 1 if shape_ok else 0

        all_details.append({
            "question": q,
            "reference": r,
            "model_prediction": p,
            "parses": int(parses),
            "shape_match": int(shape_ok),
            "parse_error": pred_err if not parses else None,
            "rate_score": score,
            "success": 1,
        })

    all_scores = [d["rate_score"] for d in all_details]
    avg_score = sum(all_scores) / len(all_scores) * 100 if all_scores else 0.0
    parse_rate = sum(d["parses"] for d in all_details) / len(all_details) if all_details else 0.0

    return {"judge_score": avg_score, "success_rate": parse_rate}, all_details
