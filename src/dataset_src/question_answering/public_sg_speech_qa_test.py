from dataset_src.base_dataset import BaseDatasetProcessor


class public_sg_speech_qa_test_dataset(BaseDatasetProcessor):
    name = "public-sg-speech"
    task_type = "Question Answering"
    sub_task = "QA (Text Instruction + Audio Context)"
    language = "EN_SG"
    metrics = "flow_judge"
