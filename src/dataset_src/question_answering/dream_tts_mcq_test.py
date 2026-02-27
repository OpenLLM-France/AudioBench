from dataset_src.base_dataset import BaseDatasetProcessor


class dream_tts_mcq_test_dataset(BaseDatasetProcessor):
    task_type = "Question Answering"
    sub_task = "MCQ (Text Instruction + Audio Context)"
    judge_binary = True
    language = "EN"
    metrics = "flow_judge"

    def _get_instruction(self, sample):
        return 'Question:\n' + sample['instruction'] + '\n Choices:\n' + sample['choices']
