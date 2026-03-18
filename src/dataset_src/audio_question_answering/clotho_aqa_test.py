from dataset_src.base_dataset import BaseDatasetProcessor


class clotho_aqa_test_dataset(BaseDatasetProcessor):
    name = "Clotho-AQA"
    task_type = "Audio Question Answering"
    language = "EN"
    metrics = "flow_judge"
