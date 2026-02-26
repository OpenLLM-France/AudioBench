from dataset_src.base_dataset import BaseDatasetProcessor

st_instructions = [
    "Listen to the speech clip and translate it into Chinese.",
    "Play the speech recording and translate it to Chinese.",
    "Hear the audio clip and convert it to Chinese.",
    "Listen to the speech and translate it into Chinese.",
    "Play the recorded speech and provide a translation in Chinese.",
    "Hear the speech audio and translate it into Chinese.",
    "Listen to the speech audio and translate it to Chinese.",
    "Play the speech clip and translate it into Chinese.",
    "Listen to the speech recording and translate it into Chinese.",
    "Hear the audio clip and translate it to Chinese.",
    "Listen to the speech and provide a translation in Chinese.",
    "Play the audio of the speech and translate it into Chinese.",
    "Hear the speech and convert it to Chinese.",
    "Listen to the speech audio clip and translate it to Chinese.",
    "Play the speech recording and translate it into Chinese.",
    "Listen to the speech clip and provide a translation in Chinese.",
    "Hear the recorded speech and translate it to Chinese.",
    "Play the audio speech and translate it into Chinese.",
    "Listen to the speech and translate it to Chinese.",
    "Hear the audio of the speech and translate it into Chinese."
]


class covost2_en_zh_test_dataset(BaseDatasetProcessor):
    instructions = st_instructions
    task_type = "ST-EN-ZH"
    language = "EN-ZH"
    metrics = "bleu"
