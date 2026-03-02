import logging
import os

logger = logging.getLogger(__name__)


def load_model(model_name, backend="transformers", model_path=None):
    """Factory: return a BaseModel subclass, loaded and ready to generate."""
    if model_path:
        model_path  = model_path.replace("<MODELS_FOLDER>", os.getenv('MODELS'))
    if model_name == "cascade_whisper_large_v3_llama_3_8b_instruct":
        from model_src.transformers.whisper_large_v3_with_llama_3_8b_instruct import WhisperLargeV3WithLlama38BInstruct
        model = WhisperLargeV3WithLlama38BInstruct()

    elif model_name == "cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct":
        from model_src.transformers.whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct import WhisperLargeV2Gemma29BCptSeaLionV3Instruct
        model = WhisperLargeV2Gemma29BCptSeaLionV3Instruct()

    elif model_name == "qwen2-audio-7b-instruct":
        from model_src.vllm.qwen2_audio_7b_instruct import Qwen2Audio7BInstruct
        model = Qwen2Audio7BInstruct()

    elif model_name == "salmonn_7b":
        from model_src.transformers.salmonn_7b import Salmonn7B
        model = Salmonn7B()

    elif model_name == 'wavllm_fairseq':
        from model_src.transformers.wavllm_fairseq import WavLLMFairseq
        model = WavLLMFairseq()

    elif model_name == 'qwen-audio-chat':
        from model_src.transformers.qwen_audio_chat import QwenAudioChat
        model = QwenAudioChat()

    elif model_name == 'meralion-audiollm-whisper-sea-lion':
        from model_src.transformers.meralion_audiollm_whisper_sea_lion import MeralionAudioLLMWhisperSeaLion
        model = MeralionAudioLLMWhisperSeaLion()

    elif model_name == 'gemini-1.5-flash':
        from model_src.api.gemini_1_5_flash import Gemini15Flash
        model = Gemini15Flash()

    elif model_name == 'gemini-2-flash':
        from model_src.api.gemini_2_flash import Gemini2Flash
        model = Gemini2Flash()

    elif model_name == 'whisper_large_v3':
        from model_src.vllm.whisper_large_v3 import WhisperLargeV3
        model = WhisperLargeV3()

    elif model_name == 'whisper_large_v2':
        from model_src.vllm.whisper_large_v2 import WhisperLargeV2
        model = WhisperLargeV2()

    elif model_name == 'gpt-4o-audio':
        from model_src.api.gpt_4o_audio import GPT4oAudio
        model = GPT4oAudio()

    elif model_name == 'phi_4_multimodal_instruct':
        from model_src.vllm.phi_4_multimodal_instruct import Phi4MultimodalInstruct
        model = Phi4MultimodalInstruct()

    elif model_name == 'seallms_audio_7b':
        from model_src.vllm.seallms_audio_7b import SeallmsAudio7B
        model = SeallmsAudio7B()

    elif model_name.startswith('luciole_audio'):
        from model_src.nemo.luciole_audio import LucioleAudio
        model = LucioleAudio(model_path=model_path)

    elif model_name == 'canary_qwen':
        from model_src.nemo.canary_qwen import CanaryQwen
        model = CanaryQwen()

    elif model_name == 'audio_flamingo':
        from model_src.vllm.audio_flamingo import AudioFlamingo
        model = AudioFlamingo()

    elif model_name.startswith('qwen2_omni'):
        from model_src.vllm.qwen_omni import Qwen2Omni
        if model_name == 'qwen2_omni-7b':
            model = Qwen2Omni(model_path="Qwen/Qwen2.5-Omni-7B")
        else:
            model = Qwen2Omni()

    elif model_name.startswith('voxtral'):
        from model_src.transformers.mistralai_voxtral import Voxtral
        model = Voxtral()

    elif model_name == 'kimi-audio-7b-instruct':
        from model_src.transformers.kimi_audio_7b_instruct import KimiAudio7BInstruct
        model = KimiAudio7BInstruct(model_path=model_path)

    else:
        raise NotImplementedError(f"Model {model_name} not implemented yet")

    model.model_name = model_name
    model.backend = backend

    if backend == "vllm":
        if not model.supports_vllm:
            logger.warning(
                f"VLLM backend not supported for model '{model_name}'. "
                f"Falling back to transformers backend."
            )
            model.backend = "transformers"
            model.load()
        else:
            model.load_vllm()
    else:
        model.load()

    logger.info(f"Loaded model: {model_name} (backend: {backend})")
    logger.info("= = "*20)
    return model
