from dataset_src.base_dataset import BaseDatasetProcessor

gr_instructions = [
    "Can you tell the speaker's gender from the audio (Male or Female)?",
    "Can you identify the speaker's gender based on the audio (Male or Female)?",
    "From the audio, can you guess the speaker's gender (Male or Female)?",
    "Can you determine the gender of the speaker from the audio (Male or Female)?",
    "Based on the audio, can you identify the speaker's gender (Male or Female)?",
    "Can you figure out the speaker's gender from the audio (Male or Female)?",
    "Can you discern the speaker's gender based on the audio (Male or Female)?",
    "From the audio, can you determine the speaker's gender (Male or Female)?",
    "Can you recognize the speaker's gender from the audio (Male or Female)?",
    "Can you guess the gender of the speaker based on the audio (Male or Female)?"
]


class voxceleb_gender_test_dataset(BaseDatasetProcessor):
    instructions = gr_instructions
    task_type = "GR"
    judge_binary = True
    metrics = "llama3_70b_judge"

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
