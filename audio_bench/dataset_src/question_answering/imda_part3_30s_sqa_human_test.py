from audio_bench.dataset_src.base_dataset import BaseDatasetProcessor


class imda_part3_30s_sqa_human_test_dataset(BaseDatasetProcessor):
    name = "imda_part3_30s_sqa_human_test"
    task_type = "Question Answering"
    sub_task = "QA (Text Instruction + Audio Context)"
    language = "EN_SG"
    metrics = "flow_judge"
