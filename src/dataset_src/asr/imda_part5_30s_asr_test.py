from dataset_src.base_dataset import BaseDatasetProcessor
from dataset_src.prompts.prompts import asr_instructions

class imda_part5_30s_asr_test_dataset(BaseDatasetProcessor):
    name = "imda_part5_30s_asr_test"
    instructions = asr_instructions
    task_type = "ASR"
    sub_task = "Conversation"
    language = "EN_SG"
    metrics = "wer"
