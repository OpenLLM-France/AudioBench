from dataset_src.base_dataset import BaseDatasetProcessor

er_instructions = [
    "How do you perceive the speaker's emotional state from their speech (frustration, anger, excited, neutral, happiness, surprise, sad)?",
    "What emotions do you detect in the speaker's voice (frustration, anger, excited, neutral, happiness, surprise, sad)?",
    "Can you identify the speaker's emotional state from their speech (frustration, anger, excited, neutral, happiness, surprise, sad)?",
    "Based on their speech, how would you describe the speaker's emotions (frustration, anger, excited, neutral, happiness, surprise, sad)?",
    "What emotional cues can you pick up from the speaker's speech (frustration, anger, excited, neutral, happiness, surprise, sad)?",
    "How would you describe the emotions conveyed in the speaker's voice (frustration, anger, excited, neutral, happiness, surprise, sad)?",
    "What do you think the speaker is feeling based on their speech (frustration, anger, excited, neutral, happiness, surprise, sad)?",
    "Can you interpret the emotions in the speaker's speech (frustration, anger, excited, neutral, happiness, surprise, sad)?",
    "How does the speaker's speech reflect their emotional state (frustration, anger, excited, neutral, happiness, surprise, sad)?",
    "What is the emotional tone of the speaker's speech (frustration, anger, excited, neutral, happiness, surprise, sad)?"
]


class iemocap_emotion_test_dataset(BaseDatasetProcessor):
    instructions = er_instructions
    task_type = "Emotion Recognition"
    sub_task = "Emotion"
    judge_binary = True
    language = "EN"
    metrics = "flow_judge"
