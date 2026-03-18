from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.prompts.prompts import asr_instructions


class ytb_asr_batch2_dataset(BaseDatasetProcessor):
    name = "ytb_asr_batch2"
    instructions = asr_instructions
    task_type = "ASR"
    sub_task = "Youtube"
    language = "EN"
    metrics = "wer"
    audio_path = "context.audio"
    reference_path = "answer.text"
