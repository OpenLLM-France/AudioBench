from audio_bench.dataset_src.base_dataset import BaseDatasetProcessor
from audio_bench.dataset_src.prompts.prompts import asr_instructions

class common_voice_15_en_test_dataset(BaseDatasetProcessor):
    name = "common_voice_15_en_test"
    instructions = asr_instructions
    task_type = "ASR"
    sub_task = "Reading"
    language = "EN"
    metrics = "wer"
