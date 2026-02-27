from dataset_src.base_dataset import BaseDatasetProcessor


class ytb_sds_batch1_dataset(BaseDatasetProcessor):
    task_type = "Dialogue Summarization"
    audio_path = "context.audio"
    instruction_path = "instruction.text"
    reference_path = "answer.text"
    language = "EN"
    metrics = "flow_judge"
