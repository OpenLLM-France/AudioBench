from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.prompts.prompts import asr_instructions


class ytb_asr_batch1_dataset(BaseDatasetProcessor):
    instructions = asr_instructions
    task_type = "ASR"
    audio_path = "context.audio"
    reference_path = "answer.text"
