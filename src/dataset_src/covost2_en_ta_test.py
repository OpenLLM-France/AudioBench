from dataset_src.base_dataset import BaseDatasetProcessor

st_instructions = [
    "Listen to the speech clip and translate it into Tamil.",
    "Play the speech recording and translate it to Tamil.",
    "Hear the audio clip and convert it to Tamil.",
    "Listen to the speech and translate it into Tamil.",
    "Play the recorded speech and provide a translation in Tamil.",
    "Hear the speech audio and translate it into Tamil.",
    "Listen to the speech audio and translate it to Tamil.",
    "Play the speech clip and translate it into Tamil.",
    "Listen to the speech recording and translate it into Tamil.",
    "Hear the audio clip and translate it to Tamil.",
    "Listen to the speech and provide a translation in Tamil.",
    "Play the audio of the speech and translate it into Tamil.",
    "Hear the speech and convert it to Tamil.",
    "Listen to the speech audio clip and translate it to Tamil.",
    "Play the speech recording and translate it into Tamil.",
    "Listen to the speech clip and provide a translation in Tamil.",
    "Hear the recorded speech and translate it to Tamil.",
    "Play the audio speech and translate it into Tamil.",
    "Listen to the speech and translate it to Tamil.",
    "Hear the audio of the speech and translate it into Tamil."
]


class covost2_en_ta_test_dataset(BaseDatasetProcessor):
    instructions = st_instructions
    task_type = "ST-EN-TA"
    language = "EN-TA"
    metrics = "bleu"
