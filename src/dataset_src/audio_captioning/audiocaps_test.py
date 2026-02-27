from dataset_src.base_dataset import BaseDatasetProcessor

ac_instructions = [
    "Could you help me create an audio caption for the provided clip?",
    "Please help me generate an audio caption for the audio clip.",
    "Would you mind helping me produce an audio caption for the provided clip?",
    "Could you help me formulate an audio caption for this clip?",
    "Could you assist me in creating a caption for the audio clip?",
]

class audiocaps_test_dataset(BaseDatasetProcessor):
    instructions = ac_instructions
    task_type = "Audio Captioning"
    language = "EN"
    metrics = "flow_judge"
