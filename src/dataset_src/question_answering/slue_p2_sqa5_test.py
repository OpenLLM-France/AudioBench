from dataset_src.base_dataset import BaseDatasetProcessor


class slue_p2_sqa5_test_dataset(BaseDatasetProcessor):
    name = "SLUE-P2-SQA5"
    task_type = "Question Answering"
    sub_task = "QA (Text Instruction + Audio Context)"
    language = "EN"
    metrics = "flow_judge"
