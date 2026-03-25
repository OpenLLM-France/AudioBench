from audio_bench.dataset_src.base_dataset import BaseDatasetProcessor

si_instructions = [
    "Kindly adhere to the directions provided in the audio.",
    "Please comply with the instructions given in the audio clip.",
    "Please obey the instructions that were provided in the audio.",
    "Please adhere to the instructions given in the audio.",
    "Please make sure to follow the instructions provided in the audio.",
    "Please ensure you follow the directions provided in the audio.",
    "Please adhere strictly to the instructions in the audio recording.",
    "Please adhere to the guidelines provided in the audio.",
    "Please make it a point to follow the instructions from the audio.",
    "Please listen carefully and follow the instructions given in the audio."
]


class openhermes_audio_test_dataset(BaseDatasetProcessor):
    name = "OpenHermes_audio"
    instructions = si_instructions
    task_type = "Question Answering"
    sub_task = "Spoken Instruction"
    question_key = "audio_text_instruction"
    language = "EN"
    metrics = "flow_judge"

    def _process_sample(self, sample):
        base = super()._process_sample(sample)
        base["audio_text_instruction"] = sample['speech_instruction']
        return base
