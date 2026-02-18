from jiwer import compute_measures, wer

from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.text_normalizer.preprocess_text import preprocess_text_asr_code_switch_chinese
from dataset_src.prompts.prompts import asr_cn_instructions


class aishell_asr_zh_test_dataset(BaseDatasetProcessor):
    instructions = asr_cn_instructions
    task_type = "ASR-ZH"

    def _compute_wer(self, data_with_model_predictions):
        predictions = []
        references  = []
        for item in data_with_model_predictions:
            model_prediction = preprocess_text_asr_code_switch_chinese(item["model_prediction"])
            answer           = preprocess_text_asr_code_switch_chinese(item["reference"])

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
