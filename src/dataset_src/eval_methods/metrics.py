import numpy as np
import evaluate

from jiwer import compute_measures, wer
from dataset_src.text_normalizer.preprocess_text import preprocess_text_asr


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

def compute_bleu(references, predictions):
    sacrebleu = evaluate.load("sacrebleu")
    corpus_bleu = sacrebleu.compute(predictions=predictions, references=references, tokenize='flores101')['score']

    per_sample_bleus = []
    for p, r in zip(predictions, references):
        s = sacrebleu.compute(predictions=[p], references=[r], tokenize='flores101')['score']
        per_sample_bleus.append(s)

    return {"bleu": build_metric_stats(per_sample_bleus, corpus_bleu)}