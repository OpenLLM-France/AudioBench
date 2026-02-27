from dataset_src.base_dataset import BaseDatasetProcessor

st_instructions = [
    "Listen to the speech clip and translate it into Indonesian.",
    "Play the speech recording and translate it to Indonesian.",
    "Hear the audio clip and convert it to Indonesian.",
    "Listen to the speech and translate it into Indonesian.",
    "Play the recorded speech and provide a translation in Indonesian.",
    "Hear the speech audio and translate it into Indonesian.",
    "Listen to the speech audio and translate it to Indonesian.",
    "Play the speech clip and translate it into Indonesian.",
    "Listen to the speech recording and translate it into Indonesian.",
    "Hear the audio clip and translate it to Indonesian.",
    "Listen to the speech and provide a translation in Indonesian.",
    "Play the audio of the speech and translate it into Indonesian.",
    "Hear the speech and convert it to Indonesian.",
    "Listen to the speech audio clip and translate it to Indonesian.",
    "Play the speech recording and translate it into Indonesian.",
    "Listen to the speech clip and provide a translation in Indonesian.",
    "Hear the recorded speech and translate it to Indonesian.",
    "Play the audio speech and translate it into Indonesian.",
    "Listen to the speech and translate it to Indonesian.",
    "Hear the audio of the speech and translate it into Indonesian."
]


class covost2_en_id_test_dataset(BaseDatasetProcessor):
    instructions = st_instructions
    task_type = "AST"
    sub_task = "EN-ID"
    language = "EN-ID"
    metrics = "bleu"
