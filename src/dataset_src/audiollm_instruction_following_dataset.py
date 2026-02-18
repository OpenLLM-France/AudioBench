from dataset_src.base_dataset import BaseDatasetProcessor


class audiollm_instruction_following_dataset(BaseDatasetProcessor):
    task_type = None
    question_key = "text"
    reference_key = "answer"

    def __init__(self, raw_data, number_of_samples):
        if isinstance(raw_data, dict):
            raw_data = raw_data["train"]
        super().__init__(raw_data, number_of_samples)

    def _process_sample(self, sample):
        return {
            "audio": sample['context'],
            "text": sample['instruction'],
            "answer": sample['answer'],
            "dimension": sample['instruction_type'],
            "rule_type": sample['rule'],
            "rule_target": sample['rule_content'],
            "task_type": sample['instruction_type'],
        }

    def compute_score(self, data_with_model_predictions, metrics=None):
        if metrics == 'llama3_70b_judge_combined':
            questions = []
            predictions = []
            references = []
            dimensions = []
            rules = []
            rule_targets = []

            for item in data_with_model_predictions:
                questions.append(item["text"])
                references.append(item["answer"])
                predictions.append(item["model_prediction"])
                dimensions.append(item['dimension'])
                rules.append(item['rule_type'])
                rule_targets.append(item['rule_target'])

            from dataset_src.eval_methods.eval_llama3_70b_combined import llama3_70b_as_judge_binary
            llama3_70b_judge_results, all_details = llama3_70b_as_judge_binary(
                "meta-llama/Meta-Llama-3-70B-Instruct",
                [questions, references, predictions, dimensions, rules, rule_targets]
            )
            return {'llama3_70b_judge_combined': llama3_70b_judge_results, 'details': all_details}
        else:
            raise ValueError(f"Unsupported metric: {metrics}. Supported metric: 'llama3_70b_combined', 'llama3_70b_judge_binary'")
