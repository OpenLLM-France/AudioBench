from dataset_src.base_dataset import BaseDatasetProcessor


class slue_p2_sqa5_test_dataset(BaseDatasetProcessor):
    task_type = "SQA"
    language = "EN"
    metrics = "flow_judge"
