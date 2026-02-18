from dataset_src.base_dataset import BaseDatasetProcessor

ar_instructions = [
    "Can you guess the speaker's nationality from their accent?",
    "Based on the accent, can you guess the speaker's nationality?",
    "Can you identify the nationality of the speaker by their accent?",
    "From the speaker's accent, can you tell their nationality?",
    "Can you guess the nationality from the speaker's accent?",
    "Based on their accent, can you determine the speaker's nationality?",
    "Can you tell the nationality of the speaker based on their accent?",
    "From the accent, can you identify the speaker's nationality?",
    "Can you recognize the speaker's nationality from their accent?",
    "Based on the accent, can you identify the speaker's nationality?"
]


class voxceleb_accent_test_dataset(BaseDatasetProcessor):
    instructions = ar_instructions
    task_type = "AR"
    judge_binary = True

    def format_model_predictions(self, input_data, model_predictions):
        data_with_model_predictions = []
        for sample in input_data:
            new_sample = sample.copy()
            if "audio" in new_sample:
                del new_sample["audio"]
            new_sample['reference'] = new_sample['reference'].replace('From the audio, I guess the speaker is from ', '')
            new_sample['model_prediction'] = model_predictions.pop(0)
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions
