from dataset_src.base_dataset import BaseDatasetProcessor


class audiocaps_qa_test_dataset(BaseDatasetProcessor):
    task_type = "ASQA"
    metrics = "llama3_70b_judge"
