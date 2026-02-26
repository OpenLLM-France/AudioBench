import functools
import logging
import os

# add parent directory to sys.path
import sys
sys.path.append('.')

from datasets import load_dataset, load_from_disk


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# ---------------------------------------------------------------------------
# Dict: DATASET_SOURCES
#   HuggingFace datasets : (hf_path, split)  or  (hf_path, split, data_dir)
#   Disk datasets        : (disk_path,)          — tuple of length 1
#   split=None           : returns the full DatasetDict (no split selection)
# ---------------------------------------------------------------------------
DATASET_SOURCES = {
    # --- Public HuggingFace datasets (split='test') ---
    'cn_college_listen_mcq_test':       ('AudioLLMs/cn_college_listen_mcq_test', 'test'),
    'slue_p2_sqa5_test':                ('AudioLLMs/slue_p2_sqa5_test', 'test'),
    'public_sg_speech_qa_test':         ('AudioLLMs/public_sg_speech_qa_test', 'test'),
    'dream_tts_mcq_test':               ('AudioLLMs/dream_tts_mcq_test', 'test'),
    'librispeech_test_clean':           ('AudioLLMs/librispeech_test_clean', 'test'),
    'librispeech_test_other':           ('AudioLLMs/librispeech_test_other', 'test'),
    'common_voice_15_en_test':          ('AudioLLMs/common_voice_15_en_test', 'test'),
    'peoples_speech_test':              ('AudioLLMs/peoples_speech_test', 'test'),
    'gigaspeech_test':                  ('AudioLLMs/gigaspeech_test', 'test'),
    'earnings21_test':                  ('AudioLLMs/earnings21_test', 'test'),
    'earnings22_test':                  ('AudioLLMs/earnings22_test', 'test'),
    'tedlium3_test':                    ('AudioLLMs/tedlium3_test', 'test'),
    'tedlium3_long_form_test':          ('AudioLLMs/tedlium3_long_form_test', 'test'),
    'openhermes_audio_test':            ('AudioLLMs/openhermes_instruction_test', 'test'),
    'alpaca_audio_test':                ('AudioLLMs/alpaca_audio_test', 'test'),
    'audiocaps_test':                   ('AudioLLMs/audiocaps_test', 'test'),
    'wavcaps_test':                     ('AudioLLMs/wavcaps_test', 'test'),
    'clotho_aqa_test':                  ('AudioLLMs/clotho_aqa_test', 'test'),
    'audiocaps_qa_test':                ('AudioLLMs/audiocaps_qa_test', 'test'),
    'wavcaps_qa_test':                  ('AudioLLMs/wavcaps_qa_test', 'test'),
    'voxceleb_accent_test':             ('AudioLLMs/voxceleb_accent_test', 'test'),
    'voxceleb_gender_test':             ('AudioLLMs/voxceleb_gender_test', 'test'),
    'iemocap_gender_test':              ('AudioLLMs/iemocap_gender_recognition', 'test'),
    'iemocap_emotion_test':             ('AudioLLMs/iemocap_emotion_recognition', 'test'),
    'meld_sentiment_test':              ('AudioLLMs/meld_sentiment_test', 'test'),
    'meld_emotion_test':                ('AudioLLMs/meld_emotion_test', 'test'),
    'covost2_en_id_test':               ('AudioLLMs/covost2_en_id_test', 'test'),
    'covost2_en_zh_test':               ('AudioLLMs/covost2_en_zh_test', 'test'),
    'covost2_en_ta_test':               ('AudioLLMs/covost2_en_ta_test', 'test'),
    'covost2_id_en_test':               ('AudioLLMs/covost2_id_en_test', 'test'),
    'covost2_zh_en_test':               ('AudioLLMs/covost2_zh_en_test', 'test'),
    'covost2_ta_en_test':               ('AudioLLMs/covost2_ta_en_test', 'test'),
    'aishell_asr_zh_test':              ('AudioLLMs/aishell_1_zh_test', 'test'),
    'spoken_squad_test':                ('AudioLLMs/spoken_squad_test', 'test'),
    'muchomusic_test':                  ('AudioLLMs/mu_chomusic_test', 'test'),
    'seame_dev_man':                    ('AudioLLMs/seame_dev_man', 'test'),
    'seame_dev_sge':                    ('AudioLLMs/seame_dev_sge', 'test'),
    'mmau_mini':                        ('AudioLLMs/MMAU-mini', 'test'),

    # --- MERaLiON IMDA datasets (split='train', with data_dir) ---
    'imda_part1_asr_test':              ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'ASR-PART1-Test'),
    'imda_part2_asr_test':              ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'ASR-PART2-Test'),
    'imda_part3_30s_asr_test':          ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'ASR-PART3-Test'),
    'imda_part4_30s_asr_test':          ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'ASR-PART4-Test'),
    'imda_part5_30s_asr_test':          ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'ASR-PART5-Test'),
    'imda_part6_30s_asr_test':          ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'ASR-PART6-Test'),
    'imda_part3_30s_sqa_human_test':    ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'SQA-PART3-Test'),
    'imda_part4_30s_sqa_human_test':    ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'SQA-PART4-Test'),
    'imda_part5_30s_sqa_human_test':    ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'SQA-PART5-Test'),
    'imda_part6_30s_sqa_human_test':    ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'SQA-PART6-Test'),
    'imda_part3_30s_ds_human_test':     ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'SDS-PART3-Test'),
    'imda_part4_30s_ds_human_test':     ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'SDS-PART4-Test'),
    'imda_part5_30s_ds_human_test':     ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'SDS-PART5-Test'),
    'imda_part6_30s_ds_human_test':     ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'SDS-PART6-Test'),
    'imda_ar_sentence':                 ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'PQA-AR-Sentence-Test'),
    'imda_ar_dialogue':                 ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'PQA-AR-Dialogue-Test'),
    'imda_gr_sentence':                 ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'PQA-GR-Sentence-Test'),
    'imda_gr_dialogue':                 ('MERaLiON/Multitask-National-Speech-Corpus-v1', 'train', 'PQA-GR-Dialogue-Test'),

    # --- GigaSpeech2 (split='train', with data_dir) ---
    'gigaspeech2_thai':                 ('AudioLLMs/gigaspeech2-test', 'train', 'th-test'),
    'gigaspeech2_indo':                 ('AudioLLMs/gigaspeech2-test', 'train', 'id-test'),
    'gigaspeech2_viet':                 ('AudioLLMs/gigaspeech2-test', 'train', 'vi-test'),

    # --- Spoken-MQA (various splits) ---
    'spoken-mqa_short_digit':           ('amao0o0/spoken-mqa', 'short_digit'),
    'spoken-mqa_long_digit':            ('amao0o0/spoken-mqa', 'long_digit'),
    'spoken-mqa_single_step_reasoning': ('amao0o0/spoken-mqa', 'single_step_reasoning'),
    'spoken-mqa_multi_step_reasoning':  ('amao0o0/spoken-mqa', 'multi_step_reasoning'),

    # --- Private / disk datasets ---
    'ytb_asr_batch1':                   ('data/3_private_data/ytb_asr_batch1',),
    'ytb_asr_batch2':                   ('data/3_private_data/ytb_asr_batch2',),
    'ytb_sqa_batch1':                   ('data/3_private_data/ytb_sqa_batch1',),
    'ytb_sds_batch1':                   ('data/3_private_data/ytb_sds_batch1',),
    'ytb_pqa_batch1':                   ('data/3_private_data/ytb_pqa_batch1',),
    'cna_test':                         ('data/3_private_data/cna_ASR_v3',),
    'idpc_test':                        ('data/3_private_data/idpc_long_ASR_v1',),
    'parliament_test':                  ('data/3_private_data/parliament_long_ASR_v1',),
    'ukusnews_test':                    ('data/3_private_data/ukusnews_long_ASR_v1',),
    'mediacorp_test':                   ('data/3_private_data/mediacorp_long_ASR_v1',),
    'idpc_short_test':                  ('data/3_private_data/idpc_short_ASR_v1',),
    'parliament_short_test':            ('data/3_private_data/parliament_short_ASR_v1',),
    'ukusnews_short_test':              ('data/3_private_data/ukusnews_short_ASR_v1',),
    'mediacorp_short_test':             ('data/3_private_data/mediacorp_short_ASR_v1',),

    'audiollm_instructionfollowing':    ('YichenG170/AudioLLMInstructionFollowing', 'train'),
    
    # jsonl files
    "fleurs_fr_jsonl_test": (f'{os.getenv("DATA_DIR")}/nemo/asr/fr/context/FLEURS/test.jsonl',)
}

def load_jsonl(path):
    import json
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

# ---------------------------------------------------------------------------
# Internal loader
# ---------------------------------------------------------------------------
def _load_raw_data(dataset_name):
    source = DATASET_SOURCES.get(dataset_name)
    if source is None:
        return load_jsonl(dataset_name)
    if len(source) == 1:
        if source[0].endswith(".jsonl"):
            return load_jsonl(source[0]) 
        else:
            return load_from_disk(source[0])
    hf_path, split = source[0], source[1]
    kwargs = {}
    if len(source) == 3:
        kwargs['data_dir'] = source[2]
    data = load_dataset(hf_path, **kwargs)
    return data[split] if split else data


# ---------------------------------------------------------------------------
# Processor factory (if/elif with explicit imports)
# ---------------------------------------------------------------------------
def _create_processor(dataset_name, data_loader, number_of_samples, external_jsonl=False):

    if external_jsonl:
        from dataset_src.other.json_dataset import jsonl_dataset_processor
        return jsonl_dataset_processor(data_loader, number_of_samples)

    elif dataset_name == 'cn_college_listen_mcq_test':
        from dataset_src.question_answering.cn_college_listen_mcq_test import cn_college_listen_mcq_test_dataset
        return cn_college_listen_mcq_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'slue_p2_sqa5_test':
        from dataset_src.question_answering.slue_p2_sqa5_test import slue_p2_sqa5_test_dataset
        return slue_p2_sqa5_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'public_sg_speech_qa_test':
        from dataset_src.question_answering.public_sg_speech_qa_test import public_sg_speech_qa_test_dataset
        return public_sg_speech_qa_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'dream_tts_mcq_test':
        from dataset_src.question_answering.dream_tts_mcq_test import dream_tts_mcq_test_dataset
        return dream_tts_mcq_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'librispeech_test_clean':
        from dataset_src.asr.librispeech_test_clean import librispeech_test_clean_dataset
        return librispeech_test_clean_dataset(data_loader, number_of_samples)

    elif dataset_name == 'librispeech_test_other':
        from dataset_src.asr.librispeech_test_other import librispeech_test_other_dataset
        return librispeech_test_other_dataset(data_loader, number_of_samples)

    elif dataset_name == 'common_voice_15_en_test':
        from dataset_src.asr.common_voice_15_en_test import common_voice_15_en_test_dataset
        return common_voice_15_en_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'peoples_speech_test':
        from dataset_src.asr.peoples_speech_test import peoples_speech_test_dataset
        return peoples_speech_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'gigaspeech_test':
        from dataset_src.asr.gigaspeech_test import gigaspeech_test_dataset
        return gigaspeech_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'earnings21_test':
        from dataset_src.asr.earnings21_test import earnings21_test_dataset
        return earnings21_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'earnings22_test':
        from dataset_src.asr.earnings22_test import earnings22_test_dataset
        return earnings22_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'tedlium3_test':
        from dataset_src.asr.tedlium3_test import tedlium3_test_dataset
        return tedlium3_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'tedlium3_long_form_test':
        from dataset_src.asr.tedlium3_long_form_test import tedlium3_long_form_test_dataset
        return tedlium3_long_form_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'openhermes_audio_test':
        from dataset_src.spoken_instruction.openhermes_audio_test import openhermes_audio_test_dataset
        return openhermes_audio_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'alpaca_audio_test':
        from dataset_src.spoken_instruction.alpaca_audio_test import alpaca_audio_test_dataset
        return alpaca_audio_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'audiocaps_test':
        from dataset_src.audio_question_answering.audiocaps_test import audiocaps_test_dataset
        return audiocaps_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'wavcaps_test':
        from dataset_src.audio_question_answering.wavcaps_test import wavcaps_test_dataset
        return wavcaps_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'clotho_aqa_test':
        from dataset_src.audio_question_answering.clotho_aqa_test import clotho_aqa_test_dataset
        return clotho_aqa_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'audiocaps_qa_test':
        from dataset_src.audio_question_answering.audiocaps_qa_test import audiocaps_qa_test_dataset
        return audiocaps_qa_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'wavcaps_qa_test':
        from dataset_src.audio_question_answering.wavcaps_qa_test import wavcaps_qa_test_dataset
        return wavcaps_qa_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'voxceleb_accent_test':
        from dataset_src.accent_recognition.voxceleb_accent_test import voxceleb_accent_test_dataset
        return voxceleb_accent_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'voxceleb_gender_test':
        from dataset_src.gender_recognition.voxceleb_gender_test import voxceleb_gender_test_dataset
        return voxceleb_gender_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'iemocap_gender_test':
        from dataset_src.gender_recognition.iemocap_gender_test import iemocap_gender_test_dataset
        return iemocap_gender_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'iemocap_emotion_test':
        from dataset_src.emotion_recognition.iemocap_emotion_test import iemocap_emotion_test_dataset
        return iemocap_emotion_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'meld_sentiment_test':
        from dataset_src.emotion_recognition.meld_sentiment_test import meld_sentiment_test_dataset
        return meld_sentiment_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'meld_emotion_test':
        from dataset_src.emotion_recognition.meld_emotion_test import meld_emotion_test_dataset
        return meld_emotion_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'covost2_en_id_test':
        from dataset_src.ast.covost2_en_id_test import covost2_en_id_test_dataset
        return covost2_en_id_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'covost2_en_zh_test':
        from dataset_src.ast.covost2_en_zh_test import covost2_en_zh_test_dataset
        return covost2_en_zh_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'covost2_en_ta_test':
        from dataset_src.ast.covost2_en_ta_test import covost2_en_ta_test_dataset
        return covost2_en_ta_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'covost2_id_en_test':
        from dataset_src.ast.covost2_id_en_test import covost2_id_en_test_dataset
        return covost2_id_en_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'covost2_zh_en_test':
        from dataset_src.ast.covost2_zh_en_test import covost2_zh_en_test_dataset
        return covost2_zh_en_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'covost2_ta_en_test':
        from dataset_src.ast.covost2_ta_en_test import covost2_ta_en_test_dataset
        return covost2_ta_en_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'aishell_asr_zh_test':
        from dataset_src.asr.aishell_asr_zh_test import aishell_asr_zh_test_dataset
        return aishell_asr_zh_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'spoken_squad_test':
        from dataset_src.question_answering.spoken_squad_test import spoken_squad_test_dataset
        return spoken_squad_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'muchomusic_test':
        from dataset_src.music_question_answering.mu_chomusic_test import mu_chomusic_test_dataset
        return mu_chomusic_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part1_asr_test':
        from dataset_src.asr.imda_part1_asr_test import imda_part1_asr_test_dataset
        return imda_part1_asr_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part2_asr_test':
        from dataset_src.asr.imda_part2_asr_test import imda_part2_asr_test_dataset
        return imda_part2_asr_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part3_30s_asr_test':
        from dataset_src.asr.imda_part3_30s_asr_test import imda_part3_30s_asr_test_dataset
        return imda_part3_30s_asr_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part4_30s_asr_test':
        from dataset_src.asr.imda_part4_30s_asr_test import imda_part4_30s_asr_test_dataset
        return imda_part4_30s_asr_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part5_30s_asr_test':
        from dataset_src.asr.imda_part5_30s_asr_test import imda_part5_30s_asr_test_dataset
        return imda_part5_30s_asr_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part6_30s_asr_test':
        from dataset_src.asr.imda_part6_30s_asr_test import imda_part6_30s_asr_test_dataset
        return imda_part6_30s_asr_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part3_30s_sqa_human_test':
        from dataset_src.question_answering.imda_part3_30s_sqa_human_test import imda_part3_30s_sqa_human_test_dataset
        return imda_part3_30s_sqa_human_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part4_30s_sqa_human_test':
        from dataset_src.question_answering.imda_part4_30s_sqa_human_test import imda_part4_30s_sqa_human_test_dataset
        return imda_part4_30s_sqa_human_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part5_30s_sqa_human_test':
        from dataset_src.question_answering.imda_part5_30s_sqa_human_test import imda_part5_30s_sqa_human_test_dataset
        return imda_part5_30s_sqa_human_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part6_30s_sqa_human_test':
        from dataset_src.question_answering.imda_part6_30s_sqa_human_test import imda_part6_30s_sqa_human_test_dataset
        return imda_part6_30s_sqa_human_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part3_30s_ds_human_test':
        from dataset_src.dialogue_summarization.imda_part3_30s_ds_human_test import imda_part3_30s_ds_human_test_dataset
        return imda_part3_30s_ds_human_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part4_30s_ds_human_test':
        from dataset_src.dialogue_summarization.imda_part4_30s_ds_human_test import imda_part4_30s_ds_human_test_dataset
        return imda_part4_30s_ds_human_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part5_30s_ds_human_test':
        from dataset_src.dialogue_summarization.imda_part5_30s_ds_human_test import imda_part5_30s_ds_human_test_dataset
        return imda_part5_30s_ds_human_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_part6_30s_ds_human_test':
        from dataset_src.dialogue_summarization.imda_part6_30s_ds_human_test import imda_part6_30s_ds_human_test_dataset
        return imda_part6_30s_ds_human_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_ar_sentence':
        from dataset_src.accent_recognition.imda_ar_sentence import imda_ar_sentence_dataset
        return imda_ar_sentence_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_ar_dialogue':
        from dataset_src.accent_recognition.imda_ar_dialogue import imda_ar_dialogue_dataset
        return imda_ar_dialogue_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_gr_sentence':
        from dataset_src.gender_recognition.imda_gr_sentence import imda_gr_sentence_dataset
        return imda_gr_sentence_dataset(data_loader, number_of_samples)

    elif dataset_name == 'imda_gr_dialogue':
        from dataset_src.gender_recognition.imda_gr_dialogue import imda_gr_dialogue_dataset
        return imda_gr_dialogue_dataset(data_loader, number_of_samples)

    elif dataset_name == 'mmau_mini':
        from dataset_src.other.mmau_mini import mmau_mini_dataset
        return mmau_mini_dataset(data_loader, number_of_samples)

    elif dataset_name == 'gigaspeech2_thai':
        from dataset_src.asr.gigaspeech2_thai import gigaspeech2_thai_dataset
        return gigaspeech2_thai_dataset(data_loader, number_of_samples)

    elif dataset_name == 'gigaspeech2_indo':
        from dataset_src.asr.gigaspeech2_indo import gigaspeech2_indo_dataset
        return gigaspeech2_indo_dataset(data_loader, number_of_samples)

    elif dataset_name == 'gigaspeech2_viet':
        from dataset_src.asr.gigaspeech2_viet import gigaspeech2_viet_dataset
        return gigaspeech2_viet_dataset(data_loader, number_of_samples)

    elif dataset_name == 'spoken-mqa_short_digit':
        from dataset_src.question_answering.spoken_mqa import spokenmqa_dataset_arithmatic
        return spokenmqa_dataset_arithmatic(data_loader, number_of_samples)

    elif dataset_name == 'spoken-mqa_long_digit':
        from dataset_src.question_answering.spoken_mqa import spokenmqa_dataset_arithmatic
        return spokenmqa_dataset_arithmatic(data_loader, number_of_samples)

    elif dataset_name == 'spoken-mqa_single_step_reasoning':
        from dataset_src.question_answering.spoken_mqa import spokenmqa_dataset_reasoning
        return spokenmqa_dataset_reasoning(data_loader, number_of_samples)

    elif dataset_name == 'spoken-mqa_multi_step_reasoning':
        from dataset_src.question_answering.spoken_mqa import spokenmqa_dataset_reasoning
        return spokenmqa_dataset_reasoning(data_loader, number_of_samples)

    elif dataset_name == 'ytb_asr_batch1':
        from dataset_src.asr.ytb_asr_batch1 import ytb_asr_batch1_dataset
        return ytb_asr_batch1_dataset(data_loader, number_of_samples)

    elif dataset_name == 'ytb_asr_batch2':
        from dataset_src.asr.ytb_asr_batch2 import ytb_asr_batch2_dataset
        return ytb_asr_batch2_dataset(data_loader, number_of_samples)

    elif dataset_name == 'ytb_sqa_batch1':
        from dataset_src.question_answering.ytb_sqa_batch1 import ytb_sqa_batch1_dataset
        return ytb_sqa_batch1_dataset(data_loader, number_of_samples)

    elif dataset_name == 'ytb_sds_batch1':
        from dataset_src.dialogue_summarization.ytb_sds_batch1 import ytb_sds_batch1_dataset
        return ytb_sds_batch1_dataset(data_loader, number_of_samples)

    elif dataset_name == 'ytb_pqa_batch1':
        from dataset_src.question_answering.ytb_pqa_batch1 import ytb_pqa_batch1_dataset
        return ytb_pqa_batch1_dataset(data_loader, number_of_samples)

    elif dataset_name == 'seame_dev_man':
        from dataset_src.asr.seame_dev_man import seame_dev_man_dataset
        return seame_dev_man_dataset(data_loader, number_of_samples)

    elif dataset_name == 'seame_dev_sge':
        from dataset_src.asr.seame_dev_sge import seame_dev_sge_dataset
        return seame_dev_sge_dataset(data_loader, number_of_samples)

    elif dataset_name == 'cna_test':
        from dataset_src.asr.cna_test import cna_test_dataset
        return cna_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'idpc_test':
        from dataset_src.asr.idpc_test import idpc_test_dataset
        return idpc_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'parliament_test':
        from dataset_src.asr.parliament_test import parliament_test_dataset
        return parliament_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'ukusnews_test':
        from dataset_src.asr.ukusnews_test import ukusnews_test_dataset
        return ukusnews_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'mediacorp_test':
        from dataset_src.asr.mediacorp_test import mediacorp_test_dataset
        return mediacorp_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'idpc_short_test':
        from dataset_src.asr.idpc_short_test import idpc_short_test_dataset
        return idpc_short_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'parliament_short_test':
        from dataset_src.asr.parliament_short_test import parliament_short_test_dataset
        return parliament_short_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'ukusnews_short_test':
        from dataset_src.asr.ukusnews_short_test import ukusnews_short_test_dataset
        return ukusnews_short_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'mediacorp_short_test':
        from dataset_src.asr.mediacorp_short_test import mediacorp_short_test_dataset
        return mediacorp_short_test_dataset(data_loader, number_of_samples)

    elif dataset_name == 'audiollm_instructionfollowing':
        from dataset_src.other.audiollm_instruction_following_dataset import audiollm_instruction_following_dataset
        return audiollm_instruction_following_dataset(data_loader, number_of_samples)

    elif DATASET_SOURCES[dataset_name][0].endswith(".jsonl"):
        from dataset_src.other.json_dataset import jsonl_dataset_processor
        return jsonl_dataset_processor(data_loader, number_of_samples)

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def load_dataset_processor(dataset_name, number_of_samples=-1, dataset_path=None):
    """Return a dataset processor (data not loaded yet — call .load() first)."""
    if dataset_name not in DATASET_SOURCES and not dataset_name.endswith(".jsonl") and not dataset_path:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")

    loader = functools.partial(_load_raw_data, dataset_path if dataset_path else dataset_name)
    return _create_processor(dataset_name, loader, number_of_samples, True if dataset_path is not None else False)
 