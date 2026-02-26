from dataset_src.base_dataset import BaseDatasetProcessor


class imda_part4_30s_sqa_human_test_dataset(BaseDatasetProcessor):
    task_type = "SQA"
    language = "EN_SG"
    metrics = "flow_judge"
