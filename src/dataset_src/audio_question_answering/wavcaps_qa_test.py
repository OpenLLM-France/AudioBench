from dataset_src.base_dataset import BaseDatasetProcessor


class wavcaps_qa_test_dataset(BaseDatasetProcessor):
    task_type = "ASQA"
    metrics = "flow_judge"
