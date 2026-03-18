from dataset_src.base_dataset import BaseDatasetProcessor


class cn_college_listen_mcq_test_dataset(BaseDatasetProcessor):
    name = "cn_college_listen_mcq_test"
    task_type = "Question Answering"
    sub_task = "MCQ (Text Instruction + Audio Context)"
    judge_binary = True
    language = "ZH"
    metrics = "flow_judge"

    def _get_instruction(self, sample):
        return 'Question:\n' + sample['instruction'] + '\n Choices:\n' + sample['choices']
