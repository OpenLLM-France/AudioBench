import numpy as np
import evaluate

from jiwer import compute_measures, wer
from audio_bench.dataset_src.text_normalizer.preprocess_text import preprocess_text_asr


def build_metric_stats(all_scores, aggregate_score=None):
    """Build {score, std, quartiles, all_scores} from per-sample scores."""
    arr = np.array(all_scores, dtype=float)
    score = aggregate_score if aggregate_score is not None else float(np.mean(arr))
    q0, q1, q2, q3, q4 = np.percentile(arr, [0, 25, 50, 75, 100]).tolist()
    return {
        "score": float(score),
        "std": float(np.std(arr)),
        "quartiles": {"min": q0, "q1": q1, "median": q2, "q3": q3, "max": q4},
        "all_scores": [float(x) for x in all_scores],
    }

def get_predictions_and_references_lists(data_with_model_predictions):
    predictions=[]
    references=[]
    for item in data_with_model_predictions:
        model_prediction = preprocess_text_asr(item["model_prediction"])
        answer           = preprocess_text_asr(item["reference"])

        if len(model_prediction) == 0: model_prediction = "empty"
        if len(answer) == 0: answer = "empty"

        predictions.append(model_prediction)
        references.append(answer)
    return predictions, references

def compute_wer(references, predictions, compute_each_samples=True):
    total_wer = compute_measures(references, predictions)
    sample_wer = []
    per_sample_wers = []
    if compute_each_samples:
        for prediction, reference in zip(predictions, references):

            wer_score = wer(reference, prediction)
            per_sample_wers.append(wer_score)

            sample_wer_score = {
                "reference" : reference,
                "prediction": prediction,
                "wer"       : wer_score,
            }

            sample_wer.append(sample_wer_score)
    return {"wer": build_metric_stats(per_sample_wers, total_wer["wer"]), "details": sample_wer}

_TASK_GUIDANCE = {
    "DIALOGUE SUMMARIZATION": (
        "The model was asked to summarize a dialogue. A good response captures the key points "
        "and main ideas. It does not need to match the reference word-for-word; focus on whether "
        "it covers the essential information with appropriate detail."
    ),
    "AUDIO CAPTIONING": (
        "The model was asked to describe or caption an audio clip. A good response captures the "
        "key sounds and events. It does not need to match the reference word-for-word; focus on "
        "whether it describes the same audio content."
    ),
    "MUSIC CAPTIONING": (
        "The model was asked to describe or caption a music clip. A good response captures the "
        "key musical elements. It does not need to match the reference word-for-word; focus on "
        "whether it describes the same musical content."
    ),
    "QUESTION ANSWERING": (
        "The model was asked to answer a question based on audio content. A good response "
        "provides the correct answer with relevant details. It does not need to match the "
        "reference word-for-word; focus on whether it conveys the same meaning and information."
    ),
    "AUDIO QUESTION ANSWERING": (
        "The model was asked to answer a question about an audio clip. A good response "
        "provides the correct answer with relevant details. It does not need to match the "
        "reference word-for-word; focus on whether it conveys the same meaning and information."
    ),
    "MATH QUESTION ANSWERING": (
        "The model was asked to solve a math problem from spoken audio. Focus on whether "
        "the response gives the correct numerical answer."
    ),
    "MUSIC QUESTION ANSWERING": (
        "The model was asked to answer a question about music. A good response provides "
        "the correct answer. It does not need to match the reference word-for-word; focus "
        "on whether it conveys the same meaning."
    ),
    "ACCENT RECOGNITION": (
        "The model was asked to identify the speaker's accent. Focus on whether the response "
        "correctly identifies the same accent as the reference."
    ),
    "STRESS TEST": (
        "The model was given a sentence stress dectection and understanding task. Focus on whether the response correctly "
        "matches the reference answer."
    ),
    "EMOTION RECOGNITION": (
        "The model was asked to identify the emotion in speech. Focus on whether the response "
        "correctly identifies the same emotion as the reference."
    ),
    "GENDER RECOGNITION": (
        "The model was asked to identify the speaker's gender. Focus on whether the response "
        "correctly identifies the same gender as the reference."
    ),
    "AGE RECOGNITION": (
        "The model was asked to identify the speaker's age range. Focus on whether the response "
        "correctly identifies the same age range as the reference."
    ),
    "SPOKEN LANGUAGE IDENTIFICATION": (
        "The model was asked to identify the language being spoken. Focus on whether the response "
        "correctly identifies the same language as the reference."
    ),
    "SPEAKER COUNT": (
        "The model was asked to count the number of speakers. Focus on whether the response "
        "gives the same count as the reference."
    ),
}

def get_task_evaluation_context(task_type):
    """Return a task-aware context string for judge prompts, or empty string if unknown."""
    if not task_type:
        return ""
    key = task_type.upper().strip()
    guidance = _TASK_GUIDANCE.get(key)
    if guidance:
        return f"[Task Type: {key}]\n{guidance}\n"
    return f"[Task Type: {key}]\n"


def compute_bleu(references, predictions):
    sacrebleu = evaluate.load("sacrebleu")
    corpus_bleu = sacrebleu.compute(predictions=predictions, references=references, tokenize='flores101')['score']

    per_sample_bleus = []
    for p, r in zip(predictions, references):
        s = sacrebleu.compute(predictions=[p], references=[r], tokenize='flores101')['score']
        per_sample_bleus.append(s)

    return {"bleu": build_metric_stats(per_sample_bleus, corpus_bleu)}