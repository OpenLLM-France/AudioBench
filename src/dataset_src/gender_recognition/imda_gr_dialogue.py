from dataset_src.base_dataset import BaseDatasetProcessor


class imda_gr_dialogue_test_dataset(BaseDatasetProcessor):
    task_type = "Gender Recognition"
    sub_task = "Dialogue"
    judge_binary = True
    language = "EN_SG"
    metrics = "flow_judge"
