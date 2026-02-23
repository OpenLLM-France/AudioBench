from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.prompts.prompts import asr_instructions

class tedlium3_long_form_test_dataset(BaseDatasetProcessor):
    instructions = asr_instructions
    task_type = "ASR"
    language = "EN"
    metrics = "wer"
