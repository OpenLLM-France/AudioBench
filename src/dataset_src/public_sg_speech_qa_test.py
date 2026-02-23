from dataset_src.base_dataset import BaseDatasetProcessor


class public_sg_speech_qa_test_dataset(BaseDatasetProcessor):
    task_type = "SQA"
    language = "EN_SG"
    metrics = "llama3_70b_judge"
