import logging
import os

logger = logging.getLogger(__name__)


def load_model(model_name, backend="transformers", model_path=None, gpu_memory_utilization=0.4, batch_size=1, device=None):
    """Factory: return a BaseModel subclass, loaded and ready to generate."""
    logger.info(f"Loading {model_name} model (path: {model_path}).")
    if model_path:
        model_path  = model_path.replace("<MODELS_DIR>", os.getenv('MODELS_DIR'))
    if model_name == "cascade_whisper_large_v3_llama_3_8b_instruct":
        from audio_bench.model_src.transformers.whisper_large_v3_with_llama_3_8b_instruct import WhisperLargeV3WithLlama38BInstruct
        model = WhisperLargeV3WithLlama38BInstruct(device=device)

    elif model_name == "cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct":
        from audio_bench.model_src.transformers.whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct import WhisperLargeV2Gemma29BCptSeaLionV3Instruct
        model = WhisperLargeV2Gemma29BCptSeaLionV3Instruct(device=device)

    elif model_name == "qwen2_audio_7b_instruct":
        from audio_bench.model_src.vllm.qwen2_audio_7b_instruct import Qwen2Audio7BInstruct
        model = Qwen2Audio7BInstruct(gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_name == "salmonn_7b":
        from audio_bench.model_src.transformers.salmonn_7b import Salmonn7B
        model = Salmonn7B(device=device)

    elif model_name == 'wavllm_fairseq':
        from audio_bench.model_src.transformers.wavllm_fairseq import WavLLMFairseq
        model = WavLLMFairseq(device=device)

    elif model_name == 'qwen_audio_chat':
        from audio_bench.model_src.transformers.qwen_audio_chat import QwenAudioChat
        model = QwenAudioChat(device=device)

    elif model_name == 'meralion_audiollm_whisper_sea_lion':
        from audio_bench.model_src.transformers.meralion_audiollm_whisper_sea_lion import MeralionAudioLLMWhisperSeaLion
        model = MeralionAudioLLMWhisperSeaLion(device=device)

    elif model_name == 'gemini_1_5_flash':
        from audio_bench.model_src.api.gemini_1_5_flash import Gemini15Flash
        model = Gemini15Flash()

    elif model_name == 'gemini_2_flash':
        from audio_bench.model_src.api.gemini_2_flash import Gemini2Flash
        model = Gemini2Flash()

    elif model_name == 'whisper_large_v3':
        from audio_bench.model_src.vllm.whisper_large_v3 import WhisperLargeV3
        model = WhisperLargeV3(device=device)

    elif model_name == 'whisper_large_v2':
        from audio_bench.model_src.vllm.whisper_large_v2 import WhisperLargeV2
        model = WhisperLargeV2(device=device)

    elif model_name == 'gpt_4o_audio':
        from audio_bench.model_src.api.gpt_4o_audio import GPT4oAudio
        model = GPT4oAudio()

    elif model_name == 'phi_4_multimodal_instruct':
        from audio_bench.model_src.vllm.phi_4_multimodal_instruct import Phi4MultimodalInstruct
        model = Phi4MultimodalInstruct(gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_name == 'seallms_audio_7b':
        from audio_bench.model_src.vllm.seallms_audio_7b import SeallmsAudio7B
        model = SeallmsAudio7B(gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_name.startswith('luciole_audio'):
        from audio_bench.model_src.nemo.luciole_audio import LucioleAudio
        model = LucioleAudio(model_path=model_path, device=device)

    elif model_name == 'canary_qwen':
        from audio_bench.model_src.nemo.canary_qwen import CanaryQwen
        model = CanaryQwen(device=device)

    elif model_name.startswith('audio_flamingo'):
        from audio_bench.model_src.vllm.audio_flamingo import AudioFlamingo
        model = AudioFlamingo(gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_name.startswith('qwen2_omni'):
        from audio_bench.model_src.vllm.qwen_omni import Qwen2Omni
        if model_name == 'qwen2_omni_7b':
            model = Qwen2Omni(model_path="Qwen/Qwen2.5-Omni-7B", gpu_memory_utilization=gpu_memory_utilization, device=device)
        else:
            model = Qwen2Omni(gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_name.startswith('voxtral'):
        from audio_bench.model_src.vllm.mistralai_voxtral import Voxtral
        model = Voxtral(gpu_memory_utilization=gpu_memory_utilization, device=device)

    elif model_name == 'kimi_audio_7b_instruct':
        from audio_bench.model_src.transformers.kimi_audio_7b_instruct import KimiAudio7BInstruct
        model = KimiAudio7BInstruct(model_path=model_path, device=device)

    else:
        raise NotImplementedError(f"Model {model_name} not implemented yet")

    model.model_name = model_name
    model.backend = backend
    model.batch_size = batch_size

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

    logger.info(f"Loaded model: {model_name} (backend: {model.backend})")
    logger.info("= = "*20)
    return model
