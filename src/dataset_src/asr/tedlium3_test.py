from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.prompts.prompts import asr_instructions

class tedlium3_test_dataset(BaseDatasetProcessor):
    instructions = asr_instructions
    task_type = "ASR"
    sub_task = "Speech"
    language = "EN"
    metrics = "wer"
