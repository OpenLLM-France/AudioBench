from dataset_src.base_dataset import BaseDatasetProcessor


class mu_chomusic_test_dataset(BaseDatasetProcessor):
    task_type = "Music Question Answering"
    language = "EN"
    judge_binary = True
    metrics = "flow_judge"

    def _get_instruction(self, sample):
        return 'Question:\n' + sample['instruction'] + '\n Choices:\n' + sample['choices']
