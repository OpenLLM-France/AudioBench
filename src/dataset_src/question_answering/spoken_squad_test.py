from dataset_src.base_dataset import BaseDatasetProcessor


class spoken_squad_test_dataset(BaseDatasetProcessor):
    task_type = "SQA"
    language = "EN"
    metrics = "flow_judge"
