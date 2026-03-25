from audio_bench.dataset_src.base_dataset import BaseDatasetProcessor
from audio_bench.dataset_src.prompts.prompts import asr_instructions

class tedlium3_long_form_test_dataset(BaseDatasetProcessor):
    name = "tedlium3_long_form_test"
    instructions = asr_instructions
    task_type = "ASR"
    sub_task = "Speech"
    language = "EN"
    metrics = "wer"
