from dataset_src.base_dataset import BaseDatasetProcessor


class spoken_squad_test_dataset(BaseDatasetProcessor):
    task_type = "Question Answering"
    sub_task = "QA (Text Instruction + Audio Context)"
    language = "EN"
    metrics = "flow_judge"
