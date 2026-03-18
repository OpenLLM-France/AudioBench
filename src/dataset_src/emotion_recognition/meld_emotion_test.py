from dataset_src.base_dataset import BaseDatasetProcessor

er_instructions = [
    "How do you perceive the speaker's emotional state from their speech (neutral, joy, disgust, sadness, surprise, anger, fear)?",
    "What emotions do you detect in the speaker's voice (neutral, joy, disgust, sadness, surprise, anger, fear)?",
    "Can you identify the speaker's emotional state from their speech (neutral, joy, disgust, sadness, surprise, anger, fear)?",
    "Based on their speech, how would you describe the speaker's emotions (neutral, joy, disgust, sadness, surprise, anger, fear)?",
    "What emotional cues can you pick up from the speaker's speech (neutral, joy, disgust, sadness, surprise, anger, fear)?",
    "How would you describe the emotions conveyed in the speaker's voice (neutral, joy, disgust, sadness, surprise, anger, fear)?",
    "What do you think the speaker is feeling based on their speech (neutral, joy, disgust, sadness, surprise, anger, fear)?",
    "Can you interpret the emotions in the speaker's speech (neutral, joy, disgust, sadness, surprise, anger, fear)?",
    "How does the speaker's speech reflect their emotional state (neutral, joy, disgust, sadness, surprise, anger, fear)?",
    "What is the emotional tone of the speaker's speech (neutral, joy, disgust, sadness, surprise, anger, fear)?"
]


class meld_emotion_test_dataset(BaseDatasetProcessor):
    name = "MELD-emotion"
    instructions = er_instructions
    task_type = "Emotion Recognition"
    sub_task = "Emotion"
    judge_binary = True
    language = "EN"
    metrics = "flow_judge"
