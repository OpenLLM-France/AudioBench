from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.eval_methods.metrics import build_metric_stats


class audiollm_instruction_following_dataset(BaseDatasetProcessor):
    task_type = "Instruction Following"
    sub_task = "Format Following"
    question_key = "text"
    reference_key = "answer"
    language = "EN"
    metrics = "llama3_70b_judge_combined"

    def _process_sample(self, sample):
        return {
            "audio": sample['context'],
            "text": sample['instruction'],
            "answer": sample['answer'],
            "dimension": sample['instruction_type'],
            "rule_type": sample['rule'],
            "rule_target": sample['rule_content'],
            "task_type": self.task_type,
            "sub_task": self.sub_task,
            "language": self.language,
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
            correctness_scores = [d["correctness_rating"] for d in all_details]
            instruction_scores = [d["instruction_following_rating"] for d in all_details]
            success_scores = [float(d["success"]) for d in all_details]
            enriched = build_metric_stats(success_scores, llama3_70b_judge_results["success_rate"])
            enriched["judge_correctness_rating"] = build_metric_stats(correctness_scores, llama3_70b_judge_results["judge_correctness_rating"])
            enriched["judge_instruction_following_ratings"] = build_metric_stats(instruction_scores, llama3_70b_judge_results["judge_instruction_following_ratings"])
            enriched["dimensional_success_rate"] = llama3_70b_judge_results.get("dimensional_success_rate")
            return {'llama3_70b_judge_combined': enriched, 'details': all_details}
        else:
            raise ValueError(f"Unsupported metric: {metrics}. Supported metric: 'llama3_70b_combined', 'llama3_70b_judge_binary'")
