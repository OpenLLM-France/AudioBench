from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.prompts.prompts import asr_instructions

class librispeech_test_other_dataset(BaseDatasetProcessor):
    instructions = asr_instructions
    task_type = "ASR"
    sub_task = "Reading"
    language = "EN"
    metrics = "wer"
