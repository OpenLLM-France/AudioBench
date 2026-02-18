import random
import logging


class BaseDatasetProcessor:
    """Base class for all dataset processors in AudioBench.

    Subclasses configure behavior through class-level attributes and
    by overriding hook methods.

    Class Attributes:
        instructions: List of instruction prompts to randomly sample from.
            If None, instruction is taken from the dataset sample.
        task_type: Task type label (e.g., "ASR", "SQA", "ER", "ST-EN-ZH").
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

    instructions = None
    task_type = "UNKNOWN"
    judge_binary = False
    question_key = "instruction"
    reference_key = "reference"
    audio_path = None
    instruction_path = None
    reference_path = None

    def __init__(self, raw_data, number_of_samples):
        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        self.raw_data = raw_data
        if self.instructions is not None:
            self.prompt = self.instructions
        logging.info('Number of samples: {}'.format(len(self.raw_data)))

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
        }

    def prepare_model_input(self):
        input_data = []
        for sample in self.raw_data:
            input_data.append(self._process_sample(sample))

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        logging.info(random.sample(input_data, 1)[0])
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data

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
        references, predictions = get_predictions_and_references_lists(data_with_model_predictions)
        return compute_wer(references, predictions)

    def _compute_bleu(self, data_with_model_predictions):
        from dataset_src.eval_methods.metrics import compute_bleu

        predictions = []
        references = []
        for item in data_with_model_predictions:
            model_prediction = item["model_prediction"]
            answer = item[self.reference_key]
            if len(model_prediction) == 0:
                model_prediction = "empty"
            if len(answer) == 0:
                answer = "empty"
            predictions.append(model_prediction)
            references.append(answer)

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
            return {'llama3_70b_judge': results, 'details': all_details}

        elif metrics == 'gpt4o_judge':
            if self.judge_binary:
                from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge_binary
                results, all_details = gpt4o_as_judge_binary("", [questions, references, predictions])
            else:
                from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge
                results, all_details = gpt4o_as_judge("", [questions, references, predictions])
            return {'gpt4o_judge': results, 'details': all_details}

        elif metrics == 'meteor':
            import evaluate
            meteor = evaluate.load('meteor')
            meteor_results = meteor.compute(predictions=predictions, references=references)
            return {'meteor': float(meteor_results['meteor'])}

        return None

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
