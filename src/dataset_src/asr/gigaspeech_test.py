from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.prompts.prompts import asr_instructions

class gigaspeech_test_dataset(BaseDatasetProcessor):
    name = "gigaspeech_test"
    instructions = asr_instructions
    task_type = "ASR"
    sub_task = "Youtube"
    language = "EN"
    metrics = "wer"
