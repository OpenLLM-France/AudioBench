
Tasks:
- ASR (Automatic Speech Recognition)
    - sub_tasks: Reading, Conversation, Youtube, Speech, Code Switching
- AST (Automatic Speech Translation)
    - sub_tasks: EN-ID, EN-ZH, EN-TA, ID-EN, ZH-EN, TA-EN
- Question Answering
    - sub_tasks: MCQ (Text Instruction + Audio Context), QA (Text Instruction + Audio Context), PQA (Text Instruction + Audio Context), Math (Text Instruction + Audio Context)
- Emotion Recognition
    - sub_tasks: Emotion, Sentiment
- Gender Recognition
    - sub_tasks: Sentence, Dialogue
- Accent Recognition
    - sub_tasks: Sentence, Dialogue
- Language Recognition
- Audio Question Answering
- Audio Captioning
- Music Question Answering
- Music Captioning
- Music Understanding
- Dialogue Summarization
- Instruction Following
    - sub_tasks: Spoken Instruction, Format Following
- Speaker Verification
- Speaker Diarization
- Spoken Language Identification
- Stress Test
    - sub_tasks: SSD (Sentence Stress Detection), SSR (Sentence Stress Reasoning)
- Others


Supported datasets:


```

# == == == == == ASR English == == == == ==

DATASET=librispeech_test_clean
METRIC=wer

DATASET=librispeech_test_other
METRIC=wer

DATASET=common_voice_15_en_test
METRIC=wer

DATASET=peoples_speech_test
METRIC=wer

DATASET=gigaspeech_test
METRIC=wer

DATASET=tedlium3_test
METRIC=wer

DATASET=tedlium3_long_form_test
METRIC=wer

DATASET=earnings21_test
METRIC=wer

DATASET=earnings22_test
METRIC=wer




# == == == == == ASR - GigaSpeech2 (Multilingual) == == == == ==

DATASET=gigaspeech2_thai
METRIC=wer

DATASET=gigaspeech2_indo
METRIC=wer

DATASET=gigaspeech2_viet
METRIC=wer


# == == == == == ASR - Singlish == == == == ==

DATASET=imda_part1_asr_test
METRIC=wer

DATASET=imda_part2_asr_test
METRIC=wer

DATASET=imda_part3_30s_asr_test
METRIC=wer

DATASET=imda_part4_30s_asr_test
METRIC=wer

DATASET=imda_part5_30s_asr_test
METRIC=wer

DATASET=imda_part6_30s_asr_test
METRIC=wer


# == == == == == ASR - Mandarin == == == == ==

DATASET=aishell_asr_zh_test
METRIC=wer



# == == == == == AST (Automatic Speech Translation) == == == == ==

DATASET=covost2_en_id_test
METRIC=bleu

DATASET=covost2_en_zh_test
METRIC=bleu

DATASET=covost2_en_ta_test
METRIC=bleu

DATASET=covost2_id_en_test
METRIC=bleu

DATASET=covost2_zh_en_test
METRIC=bleu

DATASET=covost2_ta_en_test
METRIC=bleu



# == == == == == Question Answering == == == == ==

DATASET=cn_college_listen_mcq_test
METRIC=flow_judge

DATASET=slue_p2_sqa5_test
METRIC=flow_judge

DATASET=dream_tts_mcq_test
METRIC=flow_judge

DATASET=public_sg_speech_qa_test
METRIC=flow_judge

DATASET=spoken_squad_test
METRIC=flow_judge

# Singlish SQA

DATASET=imda_part3_30s_sqa_human_test
METRIC=flow_judge

DATASET=imda_part4_30s_sqa_human_test
METRIC=flow_judge

DATASET=imda_part5_30s_sqa_human_test
METRIC=flow_judge

DATASET=imda_part6_30s_sqa_human_test
METRIC=flow_judge

# Math QA

DATASET=spoken-mqa_short_digit
METRIC=acc

DATASET=spoken-mqa_long_digit
METRIC=acc

DATASET=spoken-mqa_single_step_reasoning
METRIC=acc

DATASET=spoken-mqa_multi_step_reasoning
METRIC=acc


# == == == == == Dialogue Summarization == == == == ==


DATASET=imda_part3_30s_ds_human_test
METRIC=flow_judge

DATASET=imda_part4_30s_ds_human_test
METRIC=flow_judge

DATASET=imda_part5_30s_ds_human_test
METRIC=flow_judge

DATASET=imda_part6_30s_ds_human_test
METRIC=flow_judge


# == == == == == Instruction Following == == == == ==

DATASET=openhermes_audio_test
METRIC=flow_judge

DATASET=alpaca_audio_test
METRIC=flow_judge

DATASET=audiollm_instructionfollowing
METRIC=llama3_70b_judge_combined


# == == == == == Audio Question Answering == == == == ==

DATASET=clotho_aqa_test
METRIC=flow_judge

DATASET=wavcaps_qa_test
METRIC=flow_judge

DATASET=audiocaps_qa_test
METRIC=flow_judge




# == == == == == Audio Captioning == == == == ==

DATASET=wavcaps_test
METRIC=flow_judge

DATASET=wavcaps_test
METRIC=meteor

DATASET=audiocaps_test
METRIC=flow_judge

DATASET=audiocaps_test
METRIC=meteor



# == == == == == Emotion Recognition == == == == ==

DATASET=iemocap_emotion_test
METRIC=flow_judge

DATASET=meld_sentiment_test
METRIC=flow_judge

DATASET=meld_emotion_test
METRIC=flow_judge




# == == == == == Accent Recognition == == == == ==

DATASET=voxceleb_accent_test
METRIC=flow_judge

DATASET=imda_ar_sentence
METRIC=flow_judge

DATASET=imda_ar_dialogue
METRIC=flow_judge

# == == == == == Gender Recognition == == == == ==

DATASET=voxceleb_gender_test
METRIC=flow_judge

DATASET=iemocap_gender_test
METRIC=flow_judge

DATASET=imda_gr_sentence
METRIC=flow_judge

DATASET=imda_gr_dialogue
METRIC=flow_judge

# == == == == == Music Question Answering == == == == ==

DATASET=muchomusic_test
METRIC=flow_judge

# == == == == == Audio Understanding and Reasoning (MCQ) == == == == ==

DATASET=mmau_mini
METRIC=flow_judge  # also supports: string_match, gpt4o_judge

# == == == == == ASR Code-Switching == == == == ==

# SEAME dataset for Mandarin-English code-switching with Singapore accent.
#Lyu, Dau-Cheng, Tien Ping Tan, Engsiong Chng, and Haizhou Li. "SEAME: a Mandarin-English code-switching speech corpus in south-east asia." In Interspeech, vol. 10, pp. 1986-1989. 2010.

DATASET=seame_dev_man
METRIC=wer

DATASET=seame_dev_sge
METRIC=wer


# News

```


