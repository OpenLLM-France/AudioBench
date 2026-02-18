import functools
import importlib
import logging

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
# Dict 1: DATASET_SOURCES
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
}


# ---------------------------------------------------------------------------
# Dict 2: DATASET_PROCESSORS — (module_path, class_name)
# ---------------------------------------------------------------------------
DATASET_PROCESSORS = {
    'cn_college_listen_mcq_test':       ('dataset_src.cn_college_listen_mcq_test', 'cn_college_listen_mcq_test_dataset'),
    'slue_p2_sqa5_test':                ('dataset_src.slue_p2_sqa5_test', 'slue_p2_sqa5_test_dataset'),
    'public_sg_speech_qa_test':         ('dataset_src.public_sg_speech_qa_test', 'public_sg_speech_qa_test_dataset'),
    'dream_tts_mcq_test':               ('dataset_src.dream_tts_mcq_test', 'dream_tts_mcq_test_dataset'),
    'librispeech_test_clean':           ('dataset_src.librispeech_test_clean', 'librispeech_test_clean_dataset'),
    'librispeech_test_other':           ('dataset_src.librispeech_test_other', 'librispeech_test_other_dataset'),
    'common_voice_15_en_test':          ('dataset_src.common_voice_15_en_test', 'common_voice_15_en_test_dataset'),
    'peoples_speech_test':              ('dataset_src.peoples_speech_test', 'peoples_speech_test_dataset'),
    'gigaspeech_test':                  ('dataset_src.gigaspeech_test', 'gigaspeech_test_dataset'),
    'earnings21_test':                  ('dataset_src.earnings21_test', 'earnings21_test_dataset'),
    'earnings22_test':                  ('dataset_src.earnings22_test', 'earnings22_test_dataset'),
    'tedlium3_test':                    ('dataset_src.tedlium3_test', 'tedlium3_test_dataset'),
    'tedlium3_long_form_test':          ('dataset_src.tedlium3_long_form_test', 'tedlium3_long_form_test_dataset'),
    'openhermes_audio_test':            ('dataset_src.openhermes_audio_test', 'openhermes_audio_test_dataset'),
    'alpaca_audio_test':                ('dataset_src.alpaca_audio_test', 'alpaca_audio_test_dataset'),
    'audiocaps_test':                   ('dataset_src.audiocaps_test', 'audiocaps_test_dataset'),
    'wavcaps_test':                     ('dataset_src.wavcaps_test', 'wavcaps_test_dataset'),
    'clotho_aqa_test':                  ('dataset_src.clotho_aqa_test', 'clotho_aqa_test_dataset'),
    'audiocaps_qa_test':                ('dataset_src.audiocaps_qa_test', 'audiocaps_qa_test_dataset'),
    'wavcaps_qa_test':                  ('dataset_src.wavcaps_qa_test', 'wavcaps_qa_test_dataset'),
    'voxceleb_accent_test':             ('dataset_src.voxceleb_accent_test', 'voxceleb_accent_test_dataset'),
    'voxceleb_gender_test':             ('dataset_src.voxceleb_gender_test', 'voxceleb_gender_test_dataset'),
    'iemocap_gender_test':              ('dataset_src.iemocap_gender_test', 'iemocap_gender_test_dataset'),
    'iemocap_emotion_test':             ('dataset_src.iemocap_emotion_test', 'iemocap_emotion_test_dataset'),
    'meld_sentiment_test':              ('dataset_src.meld_sentiment_test', 'meld_sentiment_test_dataset'),
    'meld_emotion_test':                ('dataset_src.meld_emotion_test', 'meld_emotion_test_dataset'),
    'covost2_en_id_test':               ('dataset_src.covost2_en_id_test', 'covost2_en_id_test_dataset'),
    'covost2_en_zh_test':               ('dataset_src.covost2_en_zh_test', 'covost2_en_zh_test_dataset'),
    'covost2_en_ta_test':               ('dataset_src.covost2_en_ta_test', 'covost2_en_ta_test_dataset'),
    'covost2_id_en_test':               ('dataset_src.covost2_id_en_test', 'covost2_id_en_test_dataset'),
    'covost2_zh_en_test':               ('dataset_src.covost2_zh_en_test', 'covost2_zh_en_test_dataset'),
    'covost2_ta_en_test':               ('dataset_src.covost2_ta_en_test', 'covost2_ta_en_test_dataset'),
    'aishell_asr_zh_test':              ('dataset_src.aishell_asr_zh_test', 'aishell_asr_zh_test_dataset'),
    'spoken_squad_test':                ('dataset_src.spoken_squad_test', 'spoken_squad_test_dataset'),
    'muchomusic_test':                  ('dataset_src.mu_chomusic_test', 'mu_chomusic_test_dataset'),
    'imda_part1_asr_test':              ('dataset_src.imda_part1_asr_test', 'imda_part1_asr_test_dataset'),
    'imda_part2_asr_test':              ('dataset_src.imda_part2_asr_test', 'imda_part2_asr_test_dataset'),
    'imda_part3_30s_asr_test':          ('dataset_src.imda_part3_30s_asr_test', 'imda_part3_30s_asr_test_dataset'),
    'imda_part4_30s_asr_test':          ('dataset_src.imda_part4_30s_asr_test', 'imda_part4_30s_asr_test_dataset'),
    'imda_part5_30s_asr_test':          ('dataset_src.imda_part5_30s_asr_test', 'imda_part5_30s_asr_test_dataset'),
    'imda_part6_30s_asr_test':          ('dataset_src.imda_part6_30s_asr_test', 'imda_part6_30s_asr_test_dataset'),
    'imda_part3_30s_sqa_human_test':    ('dataset_src.imda_part3_30s_sqa_human_test', 'imda_part3_30s_sqa_human_test_dataset'),
    'imda_part4_30s_sqa_human_test':    ('dataset_src.imda_part4_30s_sqa_human_test', 'imda_part4_30s_sqa_human_test_dataset'),
    'imda_part5_30s_sqa_human_test':    ('dataset_src.imda_part5_30s_sqa_human_test', 'imda_part5_30s_sqa_human_test_dataset'),
    'imda_part6_30s_sqa_human_test':    ('dataset_src.imda_part6_30s_sqa_human_test', 'imda_part6_30s_sqa_human_test_dataset'),
    'imda_part3_30s_ds_human_test':     ('dataset_src.imda_part3_30s_ds_human_test', 'imda_part3_30s_ds_human_test_dataset'),
    'imda_part4_30s_ds_human_test':     ('dataset_src.imda_part4_30s_ds_human_test', 'imda_part4_30s_ds_human_test_dataset'),
    'imda_part5_30s_ds_human_test':     ('dataset_src.imda_part5_30s_ds_human_test', 'imda_part5_30s_ds_human_test_dataset'),
    'imda_part6_30s_ds_human_test':     ('dataset_src.imda_part6_30s_ds_human_test', 'imda_part6_30s_ds_human_test_dataset'),
    'imda_ar_sentence':                 ('dataset_src.imda_ar_sentence', 'imda_ar_sentence_test_dataset'),
    'imda_ar_dialogue':                 ('dataset_src.imda_ar_dialogue', 'imda_ar_dialogue_test_dataset'),
    'imda_gr_sentence':                 ('dataset_src.imda_gr_sentence', 'imda_gr_sentence_test_dataset'),
    'imda_gr_dialogue':                 ('dataset_src.imda_gr_dialogue', 'imda_gr_dialogue_test_dataset'),
    'mmau_mini':                        ('dataset_src.mmau_mini', 'mmau_mini_test_dataset'),
    'gigaspeech2_thai':                 ('dataset_src.gigaspeech2_thai', 'gigaspeech2_thai_test_dataset'),
    'gigaspeech2_indo':                 ('dataset_src.gigaspeech2_indo', 'gigaspeech2_indo_test_dataset'),
    'gigaspeech2_viet':                 ('dataset_src.gigaspeech2_viet', 'gigaspeech2_viet_test_dataset'),
    'spoken-mqa_short_digit':           ('dataset_src.spoken_mqa', 'spokenmqa_dataset_arithmatic'),
    'spoken-mqa_long_digit':            ('dataset_src.spoken_mqa', 'spokenmqa_dataset_arithmatic'),
    'spoken-mqa_single_step_reasoning': ('dataset_src.spoken_mqa', 'spokenmqa_dataset_reasoning'),
    'spoken-mqa_multi_step_reasoning':  ('dataset_src.spoken_mqa', 'spokenmqa_dataset_reasoning'),
    'ytb_asr_batch1':                   ('dataset_src.ytb_asr_batch1', 'ytb_asr_batch1_dataset'),
    'ytb_asr_batch2':                   ('dataset_src.ytb_asr_batch2', 'ytb_asr_batch2_dataset'),
    'ytb_sqa_batch1':                   ('dataset_src.ytb_sqa_batch1', 'ytb_sqa_batch1_dataset'),
    'ytb_sds_batch1':                   ('dataset_src.ytb_sds_batch1', 'ytb_sds_batch1_dataset'),
    'ytb_pqa_batch1':                   ('dataset_src.ytb_pqa_batch1', 'ytb_pqa_batch1_dataset'),
    'seame_dev_man':                    ('dataset_src.seame_dev_man', 'seame_dev_man_dataset'),
    'seame_dev_sge':                    ('dataset_src.seame_dev_sge', 'seame_dev_sge_dataset'),
    'cna_test':                         ('dataset_src.cna_test', 'cna_test_dataset'),
    'idpc_test':                        ('dataset_src.idpc_test', 'idpc_test_dataset'),
    'parliament_test':                  ('dataset_src.parliament_test', 'parliament_test_dataset'),
    'ukusnews_test':                    ('dataset_src.ukusnews_test', 'ukusnews_test_dataset'),
    'mediacorp_test':                   ('dataset_src.mediacorp_test', 'mediacorp_test_dataset'),
    'idpc_short_test':                  ('dataset_src.idpc_short_test', 'idpc_short_test_dataset'),
    'parliament_short_test':            ('dataset_src.parliament_short_test', 'parliament_short_test_dataset'),
    'ukusnews_short_test':              ('dataset_src.ukusnews_short_test', 'ukusnews_short_test_dataset'),
    'mediacorp_short_test':             ('dataset_src.mediacorp_short_test', 'mediacorp_short_test_dataset'),
    'audiollm_instructionfollowing':    ('dataset_src.audiollm_instruction_following_dataset', 'audiollm_instruction_following_dataset'),
}


# ---------------------------------------------------------------------------
# Internal loader
# ---------------------------------------------------------------------------
def _load_raw_data(dataset_name):
    source = DATASET_SOURCES[dataset_name]
    if len(source) == 1:
        return load_from_disk(source[0])
    hf_path, split = source[0], source[1]
    kwargs = {}
    if len(source) == 3:
        kwargs['data_dir'] = source[2]
    data = load_dataset(hf_path, **kwargs)
    return data[split] if split else data


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def load_dataset_processor(dataset_name, number_of_samples=-1):
    """Return a BaseDatasetProcessor (data not loaded yet — call .load() first)."""
    if dataset_name not in DATASET_SOURCES:
        raise NotImplementedError("Dataset {} not implemented yet".format(dataset_name))

    loader = functools.partial(_load_raw_data, dataset_name)

    module_path, class_name = DATASET_PROCESSORS[dataset_name]
    module = importlib.import_module(module_path)
    processor_class = getattr(module, class_name)
    return processor_class(loader, number_of_samples)
