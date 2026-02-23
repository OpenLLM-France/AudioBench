from dataset_src.base_dataset import BaseDatasetProcessor


class imda_ar_dialogue_test_dataset(BaseDatasetProcessor):
    task_type = "AR"
    judge_binary = True
    language = "EN_SG"
    metrics = "llama3_70b_judge"
