import pandas as pd
from audio_bench.dataset_src.base_dataset import BaseDatasetProcessor
from audio_bench.dataset_src.eval_methods.metrics import build_metric_stats


class mmau_mini_test_dataset(BaseDatasetProcessor):
    name = "mmau_mini"
    task_type = "Audio-Understanding-Reasoning"
    language = "EN"
    judge_binary = True
    metrics = "flow_judge"

    def _get_instruction(self, sample):
        return 'Question:\n' + sample['instruction'] + '\nChoices:\n' + " ".join(sample['choices'])

    def _process_sample(self, sample):
        base = super()._process_sample(sample)
        base["task"] = sample['other_attributes']['task']
        return base

    def compute_score(self, data_with_model_predictions, metrics=None):
        questions, references, predictions = self._extract_judge_inputs(data_with_model_predictions)

        if metrics == 'llama3_70b_judge':
            from audio_bench.dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge_binary
            llama3_70b_judge_results, all_details = llama3_70b_as_judge_binary(
                "meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions]
            )
            for result, sample_other_attributes in zip(all_details, self.raw_data['other_attributes']):
                result['task'] = sample_other_attributes['task']
            df = pd.DataFrame(all_details)
            task_scores = df.groupby('task')['rate_score'].mean().to_dict()
            all_scores = [d["rate_score"] for d in all_details]
            enriched = build_metric_stats(all_scores, llama3_70b_judge_results["judge_score"])
            enriched["success_rate"] = llama3_70b_judge_results["success_rate"]
            return {'llama3_70b_judge': enriched, "task_scores": task_scores, 'details': all_details}

        elif metrics == 'string_match':
            choices = [item for item in self.raw_data['choices']]
            from audio_bench.dataset_src.eval_methods.string_match import mmau_string_match
            string_match_results, all_details = mmau_string_match([questions, references, predictions, choices])
            for result, sample_other_attributes in zip(all_details, self.raw_data['other_attributes']):
                result['task'] = sample_other_attributes['task']
            df = pd.DataFrame(all_details)
            task_scores = df.groupby('task')['rate_score'].mean().to_dict()
            all_scores = [d["rate_score"] for d in all_details]
            enriched = build_metric_stats(all_scores, string_match_results["judge_score"])
            enriched["success_rate"] = string_match_results["success_rate"]
            return {'string_match': enriched, 'task_scores': task_scores, 'details': all_details}

        elif metrics == 'gpt4o_judge':
            from audio_bench.dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge_binary
            gpt4o_judge_results, all_details = gpt4o_as_judge_binary("", [questions, references, predictions])
            for result, sample_other_attributes in zip(all_details, self.raw_data['other_attributes']):
                result['task'] = sample_other_attributes['task']
            df = pd.DataFrame(all_details)
            task_scores = df.groupby('task')['rate_score'].mean().to_dict()
            all_scores = [d["rate_score"] for d in all_details]
            enriched = build_metric_stats(all_scores, gpt4o_judge_results["judge_score"])
            enriched["success_rate"] = gpt4o_judge_results["success_rate"]
            return {'gpt4o_judge': enriched, "task_scores": task_scores, 'details': all_details}

        else:
            raise ValueError("Invalid metrics: {}".format(metrics))
