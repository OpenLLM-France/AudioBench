from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.math_utils import utils


def get_seperation_trigger(dataset: str):
    triggers = ['The answer is:', 'The answer is', 'the answer is']
    if dataset == 'gsm8k':
        triggers.append('####')
    return triggers


class spokenmqa_dataset_arithmatic(BaseDatasetProcessor):
    task_type = "Question Answering"
    sub_task = "Math (Text Instruction + Audio Context)"
    reference_key = "answer"
    language = "EN"
    metrics = "acc"

    def _process_sample(self, sample):
        return {
            "audio": sample['context'],
            "
            ": sample['context_transcript'],
            "instruction": sample['instruction']['text'],
            "answer": sample['answer']['text'],
            "task_type": self.task_type,
            "sub_task": self.sub_task,
            "language": self.language,
        }

    def format_model_predictions(self, input_data, model_predictions, llm_text_inputs=None):
        data_with_model_predictions = []
        for idx, sample in enumerate(input_data):
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            if llm_text_inputs:
                new_sample['llm_text_input'] = llm_text_inputs[idx]
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions

    def compute_score(self, data_with_model_predictions, metrics=None):
        if metrics != 'acc':
            raise ValueError(f"Unsupported metric: {metrics}. Supported metrics: 'acc' for MathQA")

        predictions = []
        references = []
        for item in data_with_model_predictions:
            if item["model_prediction"] == None:
                item["model_prediction"] = "empty"
            else:
                model_prediction = utils.answer_clean('gsm8k', get_seperation_trigger('gsm8k'), item["model_prediction"])

            answer = item["answer"]

            if not model_prediction:
                model_prediction = "empty"
            if not answer:
                answer = "empty"

            predictions.append(model_prediction)
            references.append(answer)

        details = []
        correct, wrong = 0, 0
        for prediction, reference in zip(predictions, references):
            if isinstance(reference, str):
                reference = [reference]
            if len(prediction) > 100:
                prediction = prediction[:100]
            if utils.compare_answer_with_groundtruth(prediction, *reference):
                correct += 1
            else:
                wrong += 1
            details.append({"reference": reference, "prediction": prediction})

        return {"acc": correct / (correct + wrong), "details": details}


class spokenmqa_dataset_reasoning(BaseDatasetProcessor):
    task_type = "Question Answering"
    sub_task = "Math (Text Instruction + Audio Context)"
    reference_key = "answer"
    language = "EN"
    metrics = "acc"

    def _process_sample(self, sample):
        return {
            "audio": sample['context'],
            "audio_gt": sample['context_transcript'],
            "instruction": sample['instruction']['text'],
            "answer": sample['answer']['text'],
            "task_type": self.task_type,
            "sub_task": self.sub_task,
            "language": self.language,
        }

    def format_model_predictions(self, input_data, model_predictions, llm_text_inputs=None):
        data_with_model_predictions = []
        for idx, sample in enumerate(input_data):
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            if llm_text_inputs:
                new_sample['llm_text_input'] = llm_text_inputs[idx]
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions

    def compute_score(self, data_with_model_predictions, metrics=None):
        if metrics != 'acc':
            raise ValueError(f"Unsupported metric: {metrics}. Supported metrics: 'acc' for MathQA")

        predictions = []
        references = []
        for item in data_with_model_predictions:
            if item["model_prediction"] == None:
                item["model_prediction"] = "empty"
            else:
                model_prediction = utils.answer_clean('gsm8k', get_seperation_trigger('gsm8k'), item["model_prediction"])

            answer = []
            for ans in item["answer"]:
                if "####" in ans:
                    ans = utils.delete_extra_zero(ans.split("#### ")[-1].replace(",", ""))
                ans = utils.delete_extra_zero(ans)
                answer.append(ans)

            if not model_prediction:
                model_prediction = "empty"
            if len(answer) == 0:
                answer = "empty"

            predictions.append(model_prediction)
            references.append(answer)

        details = []
        correct, wrong = 0, 0
        for prediction, reference in zip(predictions, references):
            if isinstance(reference, str):
                reference = [reference]
            if len(prediction) > 100:
                prediction = prediction[:100]
            if utils.compare_answer_with_groundtruth(prediction, *reference):
                correct += 1
            else:
                wrong += 1
            details.append({"reference": reference, "prediction": prediction})

        return {"acc": correct / (correct + wrong), "details": details}
