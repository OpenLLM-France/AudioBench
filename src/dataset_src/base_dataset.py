import random
import logging
from tqdm import tqdm

from dataset_src.eval_methods.metrics import build_metric_stats


class BaseDatasetProcessor:
    """Base class for all dataset processors in AudioBench.

    Subclasses configure behavior through class-level attributes and
    by overriding hook methods.

    Class Attributes:
        instructions: List of instruction prompts to randomly sample from.
            If None, instruction is taken from the dataset sample.
        task_type: Task type label (e.g., "ASR", "AST", "Question Answering").
        judge_binary: If True, use binary judge variants for scoring.
        question_key: Key in prediction dicts used as question for judge scoring.
        reference_key: Key in prediction dicts used as reference for scoring.
        audio_path: Dot-separated path to audio in sample.
            None means sample['context'].
            "context.audio" means sample['context']['audio'].
        instruction_path: Dot-separated path to instruction in sample.
            Only used when instructions is None.
            None means sample['instruction'].
            "instruction.text" means sample['instruction']['text'].
        reference_path: Dot-separated path to reference in sample.
            None means sample['answer'].
            "answer.text" means sample['answer']['text'].
    """

    name = None
    instructions = None
    task_type = "UNKNOWN"
    sub_task = None
    language = "UNKNOWN"
    judge_binary = False
    question_key = "instruction"
    reference_key = "reference"
    audio_path = None
    instruction_path = None
    reference_path = None
    metrics = None

    def __init__(self, data_loader, number_of_samples, min_audio_duration=None, max_audio_duration=None, ignore_offsets=False, name=None):
        if name is not None:
            self.name = name
        self._data_loader = data_loader
        self._number_of_samples = number_of_samples
        self._min_audio_duration = min_audio_duration
        self._max_audio_duration = max_audio_duration
        self._ignore_offsets = ignore_offsets
        self._dataset_size = None
        if self.instructions is not None:
            self.prompt = self.instructions

    def _load_size(self):
        """Load only the dataset size without processing samples."""
        raw_data = self._data_loader()
        self._dataset_size = len(raw_data)
        return self._dataset_size

    def load(self):
        """Actually load the raw data. Call before prepare_model_input()."""
        raw_data = self._data_loader()
        logging.info(f"Loaded {len(raw_data)} samples")
        self._dataset_size = len(raw_data)

        if self._number_of_samples != -1:
            if self._number_of_samples > len(raw_data):
                self._number_of_samples = len(raw_data)
                logging.info(f"Requested samples exceed available. Using {self._number_of_samples}")
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(self._number_of_samples))

        logging.info(f'Number of samples: {len(raw_data)}')
        input_data = []
        for sample in tqdm(raw_data, desc="Processing samples"):
            input_data.append(self._process_sample(sample))

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        logging.info(random.sample(input_data, 1)[0])
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data

    @staticmethod
    def _resolve_path(sample, path):
        """Navigate nested dict via dot-separated path string."""
        obj = sample
        for key in path.split('.'):
            obj = obj[key]
        return obj

    def _get_audio(self, sample):
        if self.audio_path is not None:
            return self._resolve_path(sample, self.audio_path)
        return sample['context']

    def _get_instruction(self, sample):
        if self.instructions is not None:
            return random.choice(self.instructions)
        if self.instruction_path is not None:
            return self._resolve_path(sample, self.instruction_path)
        return sample['instruction']

    def _get_reference(self, sample):
        if self.reference_path is not None:
            return self._resolve_path(sample, self.reference_path)
        return sample['answer']

    def _process_sample(self, sample):
        """Build one input dict from a raw sample. Override to add extra fields."""
        return {
            "audio": self._get_audio(sample),
            "instruction": self._get_instruction(sample),
            "reference": self._get_reference(sample),
            "task_type": self.task_type,
            "sub_task": self.sub_task,
            "language": self.language,
        }

    def format_model_predictions(self, input_data, model_predictions):
        data_with_model_predictions = []
        for sample in input_data:
            new_sample = sample.copy()
            if "audio" in new_sample:
                del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions

    def _extract_judge_inputs(self, data_with_model_predictions):
        """Extract (questions, references, predictions) lists for judge scoring."""
        questions = []
        references = []
        predictions = []
        for item in data_with_model_predictions:
            questions.append(item[self.question_key])
            references.append(item[self.reference_key])
            predictions.append(item["model_prediction"])
        return questions, references, predictions

    def _compute_wer(self, data_with_model_predictions):
        from dataset_src.eval_methods.metrics import compute_wer, get_predictions_and_references_lists
        predictions, references = get_predictions_and_references_lists(data_with_model_predictions)
        return compute_wer(references, predictions)

    def _compute_bleu(self, data_with_model_predictions):
        from dataset_src.eval_methods.metrics import compute_bleu, get_predictions_and_references_lists
        predictions, references = get_predictions_and_references_lists(data_with_model_predictions)
        return compute_bleu(references, predictions)

    def _compute_judge(self, data_with_model_predictions, metrics):
        questions, references, predictions = self._extract_judge_inputs(data_with_model_predictions)

        if metrics == 'llama3_70b_judge':
            if self.judge_binary:
                from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge_binary
                results, all_details = llama3_70b_as_judge_binary(
                    "meta-llama/Meta-Llama-3-70B-Instruct",
                    [questions, references, predictions]
                )
            else:
                from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge
                results, all_details = llama3_70b_as_judge(
                    "meta-llama/Meta-Llama-3-70B-Instruct",
                    [questions, references, predictions]
                )
            return self._enrich_judge('llama3_70b_judge', results, all_details)

        elif metrics == 'gpt4o_judge':
            if self.judge_binary:
                from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge_binary
                results, all_details = gpt4o_as_judge_binary("", [questions, references, predictions])
            else:
                from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge
                results, all_details = gpt4o_as_judge("", [questions, references, predictions])
            return self._enrich_judge('gpt4o_judge', results, all_details)

        elif metrics == 'flow_judge':
            if self.judge_binary:
                from dataset_src.eval_methods.eval_flow_judge import flow_judge_as_judge_binary
                results, all_details = flow_judge_as_judge_binary("", [questions, references, predictions])
            else:
                from dataset_src.eval_methods.eval_flow_judge import flow_judge_as_judge
                results, all_details = flow_judge_as_judge("", [questions, references, predictions])
            return self._enrich_judge('flow_judge', results, all_details)

        elif metrics == 'flow_judge_api':
            if self.judge_binary:
                from dataset_src.eval_methods.eval_flow_judge_api import flow_judge_api_as_judge_binary
                results, all_details = flow_judge_api_as_judge_binary("", [questions, references, predictions])
            else:
                from dataset_src.eval_methods.eval_flow_judge_api import flow_judge_api_as_judge
                results, all_details = flow_judge_api_as_judge("", [questions, references, predictions])
            return self._enrich_judge('flow_judge_api', results, all_details)

        elif metrics == 'linagora_api_oss120':
            if self.judge_binary:
                from dataset_src.eval_methods.eval_linagora_api_oss120 import gpt4o_as_judge_binary
                results, all_details = gpt4o_as_judge_binary("", [questions, references, predictions])
            else:
                from dataset_src.eval_methods.eval_linagora_api_oss120 import gpt4o_as_judge
                results, all_details = gpt4o_as_judge("", [questions, references, predictions])
            return self._enrich_judge('linagora_api_oss120', results, all_details)

        elif metrics == 'meteor':
            import evaluate
            meteor = evaluate.load('meteor')
            corpus_meteor = float(meteor.compute(predictions=predictions, references=references)['meteor'])
            per_sample_meteors = []
            for p, r in zip(predictions, references):
                s = float(meteor.compute(predictions=[p], references=[r])['meteor'])
                per_sample_meteors.append(s)
            return {'meteor': build_metric_stats(per_sample_meteors, corpus_meteor)}

        return None

    @staticmethod
    def _enrich_judge(metric_name, results, all_details):
        """Wrap judge results with per-sample statistics."""
        all_scores = [d["rate_score"] for d in all_details]
        enriched = build_metric_stats(all_scores, results["judge_score"])
        enriched["success_rate"] = results["success_rate"]
        return {metric_name: enriched, 'details': all_details}

    def compute_score(self, data_with_model_predictions, metrics=None):
        if metrics == 'wer':
            return self._compute_wer(data_with_model_predictions)

        elif metrics == 'bleu':
            return self._compute_bleu(data_with_model_predictions)

        else:
            result = self._compute_judge(data_with_model_predictions, metrics)
            if result is not None:
                return result
            raise ValueError("Invalid metrics: {}".format(metrics))
