from audio_bench.dataset_src.base_dataset import BaseDatasetProcessor


class audiocaps_qa_test_dataset(BaseDatasetProcessor):
    name = "AudioCaps-QA"
    task_type = "Audio Question Answering"
    language = "EN"
    metrics = "flow_judge"
