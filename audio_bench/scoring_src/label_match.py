"""Deterministic label matching for closed-set classification tasks.

Replaces the LLM judge on tasks whose answer is one of a small fixed set of
labels (gender, emotion, spoken language, age). The reference and the model
prediction are both reduced to a canonical label by keyword extraction
(accent-insensitive, EN + FR), and the sample scores 1.0 iff they match.

Validated against the binary flow_judge on existing results: per-sample
agreement 94-99 %, model ranking preserved. Notably the judge *under-credited*
correct answers, so this metric is also more consistent.

Per-sample score is 0/1; the reported aggregate is mean*100 (0-100), matching
the binary judge scale so existing consumers stay unchanged.

Emotion recognition is hedge-tolerant: a prediction naming at most two
emotions scores 1.0 if the reference is among them ("frustration or anger"
matches "anger"), since these models often name overlapping emotions. Listing
three or more emotions does not earn credit, so dumping every label can't game
the score.

Age recognition uses a slightly fuzzier label space: precise decade buckets
("teens", "20".."80") when the text names a decade or numeric range, with a
fallback to coarse descriptors ("young", "middle", "old", "adult"). Validated
at 95.5 % per-sample agreement and rho 0.97 on model ranking vs the judge.
"""
import re
import unicodedata

from audio_bench.scoring_src.metrics import build_metric_stats

# Minimum gold samples a class needs to enter the macro-F1 / macro-recall
# average. Classes below this are too noisy to estimate per-class F1 (a single
# sample would otherwise carry 1/n_classes of the macro weight). They are still
# reported in `per_class`, just excluded from the headline macro aggregates.
MACRO_MIN_SUPPORT = 10


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


def _emotion_labels(text):
    """All distinct emotions named, ordered by first appearance in the text.

    Lets a hedged prediction ("frustration or anger") be matched against any
    of the emotions it mentions, rather than only the first one by _EMO order.
    """
    t = _norm(text)
    hits = []
    for label, pat in _EMO:
        m = re.search(pat, t)
        if m:
            hits.append((m.start(), label))
    return [label for _, label in sorted(hits)]


# Precise decade buckets, matched first. EN words + FR "-aine"/"-naire" forms.
_AGE_DECADE = [
    ("teens", r"teen|adolescen|\bados?\b|dizaine"),
    ("20", r"twent|vingtaine|vingtenaire"),
    ("30", r"thirt|trentaine|trentenaire"),
    ("40", r"fort(?:y|ies)|quarantaine|quadragenaire"),
    ("50", r"fift|cinquantaine|quinquagenaire"),
    ("60", r"sixt|soixantaine|sexagenaire"),
    ("70", r"sevent|septuagenaire|soixante-dix"),
    ("80", r"eight(?:y|ies)|octogenaire|quatre-vingt"),
]
# Coarse descriptors, only used when no decade/number is present. Order matters:
# "young adult" -> young (not adult), "middle-aged" -> middle (not old/aged).
_AGE_COARSE = [
    ("young", r"young|jeune"),
    ("middle", r"middle.?aged|middle|moyen"),
    ("old", r"\bold\b|older|elderly|senior|agee|vieil|vieux"),
    ("adult", r"adult"),
]


def _age(text):
    t = _norm(text)
    for label, pat in _AGE_DECADE:
        if re.search(pat, t):
            return label
    # "30s", "20s"
    m = re.search(r"\b([1-8]0)s\b", t)
    if m:
        d = int(m.group(1))
        return "teens" if d < 20 else str(d)
    # numeric ranges: 13-19, 20-29 -> a decade; wide spans (e.g. 20-60) are coarse
    m = re.search(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b", t)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi - lo > 12:
            return "adult"
        return "teens" if lo < 20 else str((lo // 10) * 10)
    for label, pat in _AGE_COARSE:
        if re.search(pat, t):
            return label
    return None


def _yesno(text):
    """Closed yes/no answer (EN + FR). Used for binary verification tasks such
    as speaker identification ("is the speaker in both clips the same?"), where
    the reference is the gold yes/no for the question as asked, so a direct
    match needs no polarity handling. `no` is checked first because the French
    "non" / English "no" are the unambiguous negatives; positives never contain
    them.
    """
    t = _norm(text)
    if re.search(r"\b(no|non|nope|nan|false|faux)\b", t):
        return "no"
    if re.search(r"\b(yes|yeah|yep|yup|true|oui|ouais|vrai)\b", t):
        return "yes"
    return None


_EXTRACTORS = {
    "GENDER RECOGNITION": _gender,
    "SPOKEN LANGUAGE IDENTIFICATION": _language,
    "EMOTION RECOGNITION": _emotion,
    "AGE RECOGNITION": _age,
    "SPEAKER IDENTIFICATION": _yesno,
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

    is_emotion = key == "EMOTION RECOGNITION"

    all_scores, details = [], []
    for ref, pred in zip(references, predictions):
        ref_label = extract(ref)
        detail = {"reference": ref, "model_prediction": pred, "ref_label": ref_label}

        if is_emotion:
            # A hedge ("frustration or anger") gets credit if the reference is
            # among the emotions it names, but only for a genuine hedge (<=2
            # distinct labels) so that listing everything doesn't game the score.
            pred_labels = _emotion_labels(pred)
            pred_label = pred_labels[0] if pred_labels else None
            detail["pred_labels"] = pred_labels
        else:
            pred_label = extract(pred)
        detail["pred_label"] = pred_label

        # An unextractable reference is an extractor gap, not a model error: skip it
        # so it neither inflates nor deflates the score (logged via n_skipped).
        if ref_label is None:
            detail["rate_score"] = None
            details.append(detail)
            continue

        if is_emotion:
            matched = ref_label in pred_labels and len(pred_labels) <= 2
        else:
            matched = pred_label == ref_label
        score = 1.0 if matched else 0.0
        all_scores.append(score)
        detail["rate_score"] = score
        details.append(detail)

    n_skipped = sum(1 for d in details if d["rate_score"] is None)
    avg = (sum(all_scores) / len(all_scores) * 100) if all_scores else 0.0
    stats = build_metric_stats(all_scores, avg)
    stats["n_skipped_no_ref_label"] = n_skipped
    stats.update(_class_balanced_stats(details))
    return {"label_match": stats, "details": details}


def _class_balanced_stats(details):
    """Per-class precision/recall/F1 plus macro aggregates, robust to class
    imbalance (a model that always predicts the majority class — e.g. ~78 %
    "male" on CommonVoice — can't ride the prior, unlike plain accuracy).

    Classes are the gold (reference) label set; samples with an unresolvable
    reference are excluded. The effective predicted label honours the same
    matching as `score` (so the emotion hedge tolerance is respected), making
    per-class recall consistent with the micro-accuracy reported above.
    """
    gold = [d for d in details if d["rate_score"] is not None]
    classes = sorted({d["ref_label"] for d in gold})
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    support = {c: 0 for c in classes}
    for d in gold:
        ref = d["ref_label"]
        support[ref] += 1
        # honour hedge-tolerant matching: a correct sample counts as predicting ref
        pred_eff = ref if d["rate_score"] == 1.0 else d.get("pred_label")
        if pred_eff == ref:
            tp[ref] += 1
        elif pred_eff in fp:  # predicted another gold class
            fp[pred_eff] += 1

    per_class = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / support[c] if support[c] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"n": support[c], "precision": prec * 100,
                        "recall": rec * 100, "f1": f1 * 100}

    # Macro-average only over classes with enough support to estimate F1
    # reliably. Without this floor a singleton class (e.g. the lone Japanese
    # sample in CoVost2 lang-ID) gets 1/n_classes of the weight — ~14 % on one
    # example — and swings macro-F1 by ±10 pts on a coin flip. Rare classes are
    # still reported in `per_class`, just excluded from the headline aggregate.
    kept = [c for c in classes if support[c] >= MACRO_MIN_SUPPORT]
    if not kept:  # tiny dataset: fall back to all classes rather than report 0
        kept = classes
    excluded = [{"label": c, "n": support[c]} for c in classes if c not in kept]

    nk = len(kept)
    macro_f1 = sum(per_class[c]["f1"] for c in kept) / nk if nk else 0.0
    macro_recall = sum(per_class[c]["recall"] for c in kept) / nk if nk else 0.0
    total = sum(support.values())
    majority_baseline = max(support.values(), default=0) / total * 100 if total else 0.0
    return {
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,          # = balanced accuracy
        "majority_baseline": majority_baseline,
        "n_classes": len(classes),
        "macro_min_support": MACRO_MIN_SUPPORT,
        "macro_n_classes": nk,
        "excluded_rare_classes": excluded,
        "per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1]["n"])),
    }
