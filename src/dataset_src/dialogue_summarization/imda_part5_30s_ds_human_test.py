from dataset_src.base_dataset import BaseDatasetProcessor


class imda_part5_30s_ds_human_test_dataset(BaseDatasetProcessor):
    name = "imda_part5_30s_ds_human_test"
    task_type = "Dialogue Summarization"
    language = "EN_SG"
    metrics = "flow_judge"
