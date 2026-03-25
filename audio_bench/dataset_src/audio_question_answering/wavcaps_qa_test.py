from audio_bench.dataset_src.base_dataset import BaseDatasetProcessor


class wavcaps_qa_test_dataset(BaseDatasetProcessor):
    name = "WavCaps-QA"
    task_type = "Audio Question Answering"
    language = "EN"
    metrics = "flow_judge"
