import logging
import os

logger = logging.getLogger(__name__)


def load_model(model_id, backend="transformers", model_path=None, gpu_memory_utilization=0.4, batch_size=1, device=None):
    """Factory: return a BaseModel subclass, loaded and ready to generate."""
    logger.info(f"Loading {model_id} model (path: {model_path}).")
    if model_path:
        model_path  = model_path.replace("<MODELS_DIR>", os.getenv('MODELS_DIR'))
    if model_id == "cascade_whisper_large_v3_llama_3_8b_instruct":
        from audio_bench.model_src.transformers.whisper_large_v3_with_llama_3_8b_instruct import WhisperLargeV3WithLlama38BInstruct
        model = WhisperLargeV3WithLlama38BInstruct(device=device)

    elif model_id == "cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct":
        from audio_bench.model_src.transformers.whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct import WhisperLargeV2Gemma29BCptSeaLionV3Instruct
        model = WhisperLargeV2Gemma29BCptSeaLionV3Instruct(device=device)

    elif model_id == "qwen2_audio_7b_instruct":
        from audio_bench.model_src.vllm.qwen2_audio_7b_instruct import Qwen2Audio7BInstruct
        model = Qwen2Audio7BInstruct(gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_id == "salmonn_7b":
        from audio_bench.model_src.transformers.salmonn_7b import Salmonn7B
        model = Salmonn7B(device=device)

    elif model_id == 'wavllm_fairseq':
        from audio_bench.model_src.transformers.wavllm_fairseq import WavLLMFairseq
        model = WavLLMFairseq(device=device)

    elif model_id == 'qwen_audio_chat':
        from audio_bench.model_src.transformers.qwen_audio_chat import QwenAudioChat
        model = QwenAudioChat(device=device)

    elif model_id == 'meralion_audiollm_whisper_sea_lion':
        from audio_bench.model_src.transformers.meralion_audiollm_whisper_sea_lion import MeralionAudioLLMWhisperSeaLion
        model = MeralionAudioLLMWhisperSeaLion(device=device)

    elif model_id == 'gemini_1_5_flash':
        from audio_bench.model_src.api.gemini_1_5_flash import Gemini15Flash
        model = Gemini15Flash()

    elif model_id == 'gemini_2_flash':
        from audio_bench.model_src.api.gemini_2_flash import Gemini2Flash
        model = Gemini2Flash()

    elif model_id == 'whisper_large_v3':
        from audio_bench.model_src.vllm.whisper_large_v3 import WhisperLargeV3
        model = WhisperLargeV3(device=device)

    elif model_id == 'whisper_large_v2':
        from audio_bench.model_src.vllm.whisper_large_v2 import WhisperLargeV2
        model = WhisperLargeV2(device=device)

    elif model_id == 'gpt_4o_audio':
        from audio_bench.model_src.api.gpt_4o_audio import GPT4oAudio
        model = GPT4oAudio()

    elif model_id == 'phi_4_multimodal_instruct':
        from audio_bench.model_src.vllm.phi_4_multimodal_instruct import Phi4MultimodalInstruct
        model = Phi4MultimodalInstruct(gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_id == 'seallms_audio_7b':
        from audio_bench.model_src.vllm.seallms_audio_7b import SeallmsAudio7B
        model = SeallmsAudio7B(gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_id.startswith('luciole_audio') or model_id.startswith('linagora'):
        from audio_bench.model_src.nemo.luciole_audio import LucioleAudio
        model = LucioleAudio(model_path=model_path, device=device)

    elif model_id == 'canary_qwen':
        from audio_bench.model_src.nemo.canary_qwen import CanaryQwen
        model = CanaryQwen(device=device)

    elif model_id.startswith('audio_flamingo'):
        from audio_bench.model_src.vllm.audio_flamingo import AudioFlamingo
        if model_id == 'audio_flamingo_next':
            af_path = "nvidia/audio-flamingo-next-hf"
        else:
            af_path = "nvidia/audio-flamingo-3-hf"
        model = AudioFlamingo(model_path=af_path, gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_id.startswith('qwen2_omni'):
        from audio_bench.model_src.vllm.qwen_omni import Qwen2Omni
        if model_id == 'qwen2_omni_7b':
            model = Qwen2Omni(model_path="Qwen/Qwen2.5-Omni-7B", gpu_memory_utilization=gpu_memory_utilization, device=device)
            model.name = "Qwen/Qwen2.5-Omni-7B"
        else:
            model = Qwen2Omni(gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_id.startswith('voxtral'):
        from audio_bench.model_src.vllm.mistralai_voxtral import Voxtral
        model = Voxtral(gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_id == 'kimi_audio_7b_instruct':
        from audio_bench.model_src.transformers.kimi_audio_7b_instruct import KimiAudio7BInstruct
        model = KimiAudio7BInstruct(model_path=model_path, device=device)

    else:
        raise NotImplementedError(f"Model {model_id} not implemented yet")

    model.model_id = model_id
    if model.name is None:
        model.name = model_id
    model.backend = backend
    model.batch_size = batch_size

    if backend == "vllm":
        if not model.supports_vllm:
            logger.warning(
                f"VLLM backend not supported for model '{model_id}'. "
                f"Falling back to transformers backend."
            )
            model.backend = "transformers"
            model.load()
        else:
            model.load_vllm()
    else:
        model.load()

    logger.info(f"Loaded model: {model.name} [id={model_id}] (backend: {model.backend})")
    logger.info("= = "*20)
    return model


_MODEL_ID_TO_NAME = {
    "cascade_whisper_large_v3_llama_3_8b_instruct": "openai/whisper-large-v3 + meta-llama/Meta-Llama-3-8B-Instruct",
    "cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct": "openai/whisper-large-v2 + aisingapore/gemma2-9b-cpt-sea-lionv3-instruct",
    "qwen2_audio_7b_instruct": "Qwen/Qwen2-Audio-7B-Instruct",
    "salmonn_7b": "tsinghua-ee/SALMONN-7B",
    "wavllm_fairseq": "microsoft/WavLLM",
    "qwen_audio_chat": "Qwen/Qwen-Audio-Chat",
    "meralion_audiollm_whisper_sea_lion": "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION",
    "gemini_1_5_flash": "Google/gemini-1.5-flash",
    "gemini_2_flash": "Google/gemini-2.0-flash-exp",
    "whisper_large_v3": "openai/whisper-large-v3",
    "whisper_large_v2": "openai/whisper-large-v2",
    "gpt_4o_audio": "OpenAI/gpt-4o-audio-preview",
    "phi_4_multimodal_instruct": "microsoft/Phi-4-multimodal-instruct",
    "seallms_audio_7b": "SeaLLMs/SeaLLMs-Audio-7B",
    "canary_qwen": "nvidia/canary-qwen-2.5b",
    "kimi_audio_7b_instruct": "moonshotai/Kimi-Audio-7B-Instruct",
    "qwen2_omni_7b": "Qwen/Qwen2.5-Omni-7B",
    "qwen2_omni_3b": "Qwen/Qwen2.5-Omni-3B",
}

# Prefix-based entries (checked when exact match fails)
_MODEL_ID_PREFIX_TO_NAME = {
    "audio_flamingo_next": "nvidia/audio-flamingo-next-hf",
    "audio_flamingo": "nvidia/audio-flamingo-3-hf",
    "voxtral": "mistralai/Voxtral-Mini-3B-2507",
}


def get_model_name(model_id):
    """Return the display name for a model_id without loading the model."""
    name = _MODEL_ID_TO_NAME.get(model_id)
    if name is not None:
        return name
    for prefix, name in _MODEL_ID_PREFIX_TO_NAME.items():
        if model_id.startswith(prefix):
            return name
    return model_id
