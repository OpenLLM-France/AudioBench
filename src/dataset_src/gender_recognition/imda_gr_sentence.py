from dataset_src.base_dataset import BaseDatasetProcessor


class imda_gr_sentence_test_dataset(BaseDatasetProcessor):
    name = "imda_gr_sentence"
    task_type = "Gender Recognition"
    sub_task = "Sentence"
    judge_binary = True
    language = "EN_SG"
    metrics = "flow_judge"
