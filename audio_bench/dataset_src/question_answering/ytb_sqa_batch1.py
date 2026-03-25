from audio_bench.dataset_src.base_dataset import BaseDatasetProcessor


class ytb_sqa_batch1_dataset(BaseDatasetProcessor):
    name = "ytb_sqa_batch1"
    task_type = "Question Answering"
    sub_task = "QA (Text Instruction + Audio Context)"
    audio_path = "context.audio"
    instruction_path = "instruction.text"
    reference_path = "answer.text"
    language = "EN"
    metrics = "flow_judge"
