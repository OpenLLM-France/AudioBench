import re

from jiwer import compute_measures, wer

from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.text_normalizer.preprocess_text import preprocess_text_asr


class gigaspeech2_indo_test_dataset(BaseDatasetProcessor):
    task_type = "ASR"

    def _compute_wer(self, data_with_model_predictions):

        def merge_spaces(text):
            return re.sub(r'\s+', ' ', text).strip()

        predictions=[]
        references=[]
        for item in data_with_model_predictions:
            model_prediction = preprocess_text_asr(item["model_prediction"])
            answer           = preprocess_text_asr(item["reference"])

            #model_prediction = merge_spaces(model_prediction)
            #answer           = merge_spaces(answer)

            if len(model_prediction) == 0: model_prediction = "empty"
            if len(answer) == 0: answer = "empty"

            predictions.append(model_prediction)
            references.append(answer)

        sample_wer = []
        incorrect  = 0
        total      = 0
        for prediction, reference in zip(predictions, references):
            measures   = compute_measures(reference, prediction)
            incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
            total     += measures["substitutions"] + measures["deletions"] + measures["hits"]

            wer_score = wer(reference, prediction)

            sample_wer_score = {
                "reference" : reference,
                "prediction": prediction,
                "wer"       : wer_score,
            }

            sample_wer.append(sample_wer_score)

        total_wer = incorrect / total

        return {"wer": total_wer, "sample_wer": sample_wer}
