from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.prompts.prompts import asr_instructions


class idpc_short_test_dataset(BaseDatasetProcessor):
    name = "idpc_short_test"
    instructions = asr_instructions
    task_type = "ASR"
    sub_task = "Conversation"
    audio_path = "context.audio"
    reference_path = "answer.text"
    language = "EN_SG"
    metrics = "wer"
