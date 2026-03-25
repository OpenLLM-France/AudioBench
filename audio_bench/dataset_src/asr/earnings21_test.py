from audio_bench.dataset_src.base_dataset import BaseDatasetProcessor
from audio_bench.dataset_src.prompts.prompts import asr_instructions

class earnings21_test_dataset(BaseDatasetProcessor):
    name = "earnings21_test"
    instructions = asr_instructions
    task_type = "ASR"
    sub_task = "Conversation"
    language = "EN"
    metrics = "wer"
