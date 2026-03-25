from audio_bench.dataset_src.base_dataset import BaseDatasetProcessor


class imda_ar_sentence_test_dataset(BaseDatasetProcessor):
    name = "imda_ar_sentence"
    task_type = "Accent Recognition"
    sub_task = "Sentence"
    judge_binary = True
    language = "EN_SG"
    metrics = "flow_judge"
