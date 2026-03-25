from audio_bench.dataset_src.base_dataset import BaseDatasetProcessor
from audio_bench.dataset_src.prompts.prompts import asr_instructions


class idpc_test_dataset(BaseDatasetProcessor):
    name = "idpc_test"
    instructions = asr_instructions
    task_type = "ASR"
    sub_task = "Conversation"
    audio_path = "context.audio"
    reference_path = "answer.text"
    language = "EN_SG"
    metrics = "wer"
