from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.prompts.prompts import asr_instructions


class cna_test_dataset(BaseDatasetProcessor):
    instructions = asr_instructions
    task_type = "ASR"
    audio_path = "context.audio"
    reference_path = "answer.text"
