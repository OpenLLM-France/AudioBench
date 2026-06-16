"""Deterministic label matching for closed-set classification tasks.

Replaces the LLM judge on tasks whose answer is one of a small fixed set of
labels (gender, emotion, spoken language). The reference and the model
prediction are both reduced to a canonical label by keyword extraction
(accent-insensitive, EN + FR), and the sample scores 1.0 iff they match.

Validated against the binary flow_judge on existing results: per-sample
agreement 94-99 %, model ranking preserved. Notably the judge *under-credited*
correct gender answers, so this metric is also more consistent.

Per-sample score is 0/1; the reported aggregate is mean*100 (0-100), matching
the binary judge scale so existing consumers stay unchanged.

Age recognition is intentionally NOT handled here: its labels are fuzzy and
open ("young", "middle-aged", "senior", decade ranges...) so it keeps the judge.
"""
import re
import unicodedata

from audio_bench.scoring_src.metrics import build_metric_stats


def _norm(text):
    """Lowercase and strip accents so EN/FR variants match one pattern."""
    text = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _gender(text):
    t = _norm(text)
    if re.search(r"\b(female|woman|women|feminine|femme|feminin\w*|elle)\b", t):
        return "female"
    if re.search(r"\b(male|man|men|masculine|homme|masculin|gentleman|boy|garcon)\b", t):
        return "male"
    return None


# (code, pattern) ordered so that more specific names win; runs on _norm() text.
_LANG = [
    ("zh", r"\b(chinese|mandarin|chinois\w*|zh)\b"),
    ("ja", r"\b(japanese|japonais\w*|ja)\b"),
    ("de", r"\b(german|deutsch|allemand\w*|de)\b"),
    ("es", r"\b(spanish|espagnol\w*|espanol|es)\b"),
    ("fr", r"\b(french|francais\w*|fr)\b"),
    ("it", r"\b(italian|italien\w*|italiano|it)\b"),
    ("en", r"\b(english|anglais\w*|en)\b"),
]


def _language(text):
    t = _norm(text).strip()
    if t in ("zh-cn", "zh_cn"):
        return "zh"
    for code, pat in _LANG:
        if re.search(pat, t):
            return code
    return None


# Canonical emotion -> synonyms (EN + FR), specific labels first to avoid overlap.
_EMO = [
    ("frustration", r"frustrat"),
    ("surprise", r"surpris|surprise|etonn|stupefa|ebaubi|abasourdi|ahuri"),
    ("fear", r"\bfear|afraid|peur|craint|effray|apprehens|anxie|terrifi"),
    ("disgust", r"disgust|degout|mepris|ecoeur|repuls|aversion"),
    ("anger", r"\banger|angry|coler|fache|furieu|agress|irrit|rage|enerv"),
    ("sad", r"\bsad\b|sadness|triste|tristesse|melancol|chagrin|deprim|abattu|morose"),
    ("excited", r"excit"),
    ("happy", r"happy|happiness|\bjoy\b|joie|joyeu|heureu|euphor|alleg|\bravi|content|gaiet|jovial"),
    ("neutral", r"neutral|neutre|neutralit"),
]


def _emotion(text):
    t = _norm(text)
    for label, pat in _EMO:
        if re.search(pat, t):
            return label
    return None


_EXTRACTORS = {
    "GENDER RECOGNITION": _gender,
    "SPOKEN LANGUAGE IDENTIFICATION": _language,
    "EMOTION RECOGNITION": _emotion,
}


def supports(task_type):
    return bool(task_type) and task_type.upper().strip() in _EXTRACTORS


def compute_label_match(references, predictions, task_type):
    """Score closed-set predictions deterministically. Mirrors the judge output shape."""
    key = (task_type or "").upper().strip()
    extract = _EXTRACTORS.get(key)
    if extract is None:
        raise ValueError(
            f"label_match does not support task '{task_type}'. "
            f"Supported: {sorted(_EXTRACTORS)}"
        )

    all_scores, details = [], []
    for ref, pred in zip(references, predictions):
        ref_label = extract(ref)
        pred_label = extract(pred)
        # An unextractable reference is an extractor gap, not a model error: skip it
        # so it neither inflates nor deflates the score (logged via n_skipped).
        if ref_label is None:
            details.append({"reference": ref, "model_prediction": pred,
                            "ref_label": None, "pred_label": pred_label, "rate_score": None})
            continue
        score = 1.0 if pred_label == ref_label else 0.0
        all_scores.append(score)
        details.append({"reference": ref, "model_prediction": pred,
                        "ref_label": ref_label, "pred_label": pred_label, "rate_score": score})

    n_skipped = sum(1 for d in details if d["rate_score"] is None)
    avg = (sum(all_scores) / len(all_scores) * 100) if all_scores else 0.0
    stats = build_metric_stats(all_scores, avg)
    stats["n_skipped_no_ref_label"] = n_skipped
    return {"label_match": stats, "details": details}
