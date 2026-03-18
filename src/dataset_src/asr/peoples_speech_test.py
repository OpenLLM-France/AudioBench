from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.prompts.prompts import asr_instructions

class peoples_speech_test_dataset(BaseDatasetProcessor):
    name = "peoples_speech_test"
    instructions = asr_instructions
    task_type = "ASR"
    sub_task = "Reading"
    language = "EN"
    metrics = "wer"
