from dataset_src.base_dataset import BaseDatasetProcessor

er_instructions = [
    "What sentiment do you sense in the speaker's voice (neutral, positive, negative)?",
    "Can you determine the speaker's sentiment from their speech (neutral, positive, negative)?",
    "How would you describe the speaker's sentiment based on their speech (neutral, positive, negative)?",
    "What sentiment signals can you hear in the speaker's speech (neutral, positive, negative)?",
    "How would you interpret the sentiment expressed in the speaker's voice (neutral, positive, negative)?",
    "What sentiment do you think the speaker is conveying through their speech (neutral, positive, negative)?",
    "Can you recognize the sentiment in the speaker's speech (neutral, positive, negative)?",
    "How does the speaker's speech indicate their sentiment (neutral, positive, negative)?",
    "What sentiment tone do you hear in the speaker's speech (neutral, positive, negative)?",
    "What sentiment is conveyed through the speaker's voice (neutral, positive, negative)?"
]


class meld_sentiment_test_dataset(BaseDatasetProcessor):
    instructions = er_instructions
    task_type = "ER"
    judge_binary = True
