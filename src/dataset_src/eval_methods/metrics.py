import evaluate

from jiwer import compute_measures, wer
from dataset_src.text_normalizer.preprocess_text import preprocess_text_asr

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
    if compute_each_samples:
        for prediction, reference in zip(predictions, references):
            measures   = compute_measures(reference, prediction)

            wer_score = wer(reference, prediction)

            sample_wer_score = {
                "reference" : reference,
                "prediction": prediction,
                "wer"       : wer_score,
            }

            sample_wer.append(sample_wer_score)
    return {"wer": total_wer["wer"], "sample_wer": sample_wer}

def compute_bleu(references, predictions):
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(predictions=predictions, references=references, tokenize='flores101')

    return {"bleu": results['score']}