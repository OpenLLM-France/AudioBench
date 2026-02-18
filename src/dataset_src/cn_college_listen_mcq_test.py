from dataset_src.base_dataset import BaseDatasetProcessor


class cn_college_listen_mcq_test_dataset(BaseDatasetProcessor):
    task_type = "SQA"
    judge_binary = True

    def _get_instruction(self, sample):
        return 'Question:\n' + sample['instruction'] + '\n Choices:\n' + sample['choices']
