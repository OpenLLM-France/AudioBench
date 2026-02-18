from dataset_src.base_dataset import BaseDatasetProcessor

st_instructions = [
    "Listen to the speech clip and translate it into English.",
    "Play the speech recording and translate it to English.",
    "Hear the audio clip and convert it to English.",
    "Listen to the speech and translate it into English.",
    "Play the recorded speech and provide a translation in English.",
    "Hear the speech audio and translate it into English.",
    "Listen to the speech audio and translate it to English.",
    "Play the speech clip and translate it into English.",
    "Listen to the speech recording and translate it into English.",
    "Hear the audio clip and translate it to English.",
    "Listen to the speech and provide a translation in English.",
    "Play the audio of the speech and translate it into English.",
    "Hear the speech and convert it to English.",
    "Listen to the speech audio clip and translate it to English.",
    "Play the speech recording and translate it into English.",
    "Listen to the speech clip and provide a translation in English.",
    "Hear the recorded speech and translate it to English.",
    "Play the audio speech and translate it into English.",
    "Listen to the speech and translate it to English.",
    "Hear the audio of the speech and translate it into English."
]


class covost2_id_en_test_dataset(BaseDatasetProcessor):
    instructions = st_instructions
    task_type = "ST-ID-EN"
