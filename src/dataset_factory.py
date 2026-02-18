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
    'cn_college_listen_mcq_test':       None,
    'slue_p2_sqa5_test':                None,
    'public_sg_speech_qa_test':         None,
    'dream_tts_mcq_test':               None,
    'librispeech_test_clean':           None,
    'librispeech_test_other':           None,
    'common_voice_15_en_test':          None,
    'peoples_speech_test':              None,
    'gigaspeech_test':                  None,
    'earnings21_test':                  None,
    'earnings22_test':                  None,
    'tedlium3_test':                    None,
    'tedlium3_long_form_test':          None,
    'openhermes_audio_test':            None,
    'alpaca_audio_test':                None,
    'audiocaps_test':                   None,
    'wavcaps_test':                     None,
    'clotho_aqa_test':                  None,
    'audiocaps_qa_test':                None,
    'wavcaps_qa_test':                  None,
    'voxceleb_accent_test':             None,
    'voxceleb_gender_test':             None,
    'iemocap_gender_test':              None,
    'iemocap_emotion_test':             None,
    'meld_sentiment_test':              None,
    'meld_emotion_test':                None,
    'covost2_en_id_test':               None,
    'covost2_en_zh_test':               None,
    'covost2_en_ta_test':               None,
    'covost2_id_en_test':               None,
    'covost2_zh_en_test':               None,
    'covost2_ta_en_test':               None,
    'aishell_asr_zh_test':              None,
    'spoken_squad_test':                None,
    'muchomusic_test':                  ('dataset_src.mu_chomusic_test', 'mu_chomusic_test_dataset'),
    'imda_part1_asr_test':              None,
    'imda_part2_asr_test':              None,
    'imda_part3_30s_asr_test':          None,
    'imda_part4_30s_asr_test':          None,
    'imda_part5_30s_asr_test':          None,
    'imda_part6_30s_asr_test':          None,
    'imda_part3_30s_sqa_human_test':    None,
    'imda_part4_30s_sqa_human_test':    None,
    'imda_part5_30s_sqa_human_test':    None,
    'imda_part6_30s_sqa_human_test':    None,
    'imda_part3_30s_ds_human_test':     None,
    'imda_part4_30s_ds_human_test':     None,
    'imda_part5_30s_ds_human_test':     None,
    'imda_part6_30s_ds_human_test':     None,
    'imda_ar_sentence':                 None,
    'imda_ar_dialogue':                 None,
    'imda_gr_sentence':                 None,
    'imda_gr_dialogue':                 None,
    'mmau_mini':                        None,
    'gigaspeech2_thai':                 None,
    'gigaspeech2_indo':                 None,
    'gigaspeech2_viet':                 None,
    'spoken-mqa_short_digit':           ('dataset_src.spoken_mqa', 'spokenmqa_dataset_arithmatic'),
    'spoken-mqa_long_digit':            ('dataset_src.spoken_mqa', 'spokenmqa_dataset_arithmatic'),
    'spoken-mqa_single_step_reasoning': ('dataset_src.spoken_mqa', 'spokenmqa_dataset_reasoning'),
    'spoken-mqa_multi_step_reasoning':  ('dataset_src.spoken_mqa', 'spokenmqa_dataset_reasoning'),
    'ytb_asr_batch1':                   None,
    'ytb_asr_batch2':                   None,
    'ytb_sqa_batch1':                   None,
    'ytb_sds_batch1':                   None,
    'ytb_pqa_batch1':                   None,
    'seame_dev_man':                    None,
    'seame_dev_sge':                    None,
    'cna_test':                         None,
    'idpc_test':                        None,
    'parliament_test':                  None,
    'ukusnews_test':                    None,
    'mediacorp_test':                   None,
    'idpc_short_test':                  None,
    'parliament_short_test':            None,
    'ukusnews_short_test':              None,
    'mediacorp_short_test':             None,
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

    value = DATASET_PROCESSORS[dataset_name]
    if value is None:
        module_path, class_name = f"dataset_src.{dataset_name}", f"{dataset_name}_dataset"
    elif len(value)==1:
        module_path, class_name = value[0], f"{dataset_name}_dataset"
    else:
        module_path, class_name = value
    module = importlib.import_module(module_path)
    processor_class = getattr(module, class_name)
    return processor_class(loader, number_of_samples)
