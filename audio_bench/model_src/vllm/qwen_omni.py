import re
import logging


from audio_bench.model_src.vllm_model import VLLMModel

logger = logging.getLogger(__name__)


def _post_process_qwen2_omni_asr(model_output):
    # Try \boxed{"..."}
    m = re.search(r'\\boxed\{"(.*?)"\}', model_output, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # Try \boxed{...}
    m = re.search(r"\\boxed\{(.+?)\}", model_output)
    if m and m.group(1).strip():
        return m.group(1).strip()

    return model_output

class Qwen2Omni(VLLMModel):

    name = "Qwen/Qwen2.5-Omni-3B"

    def __init__(self, model_path="Qwen/Qwen2.5-Omni-3B", gpu_memory_utilization=0.4, device=None):
        super().__init__(model_path=model_path, gpu_memory_utilization=gpu_memory_utilization, device=device)
        self._asr_text_processor = _post_process_qwen2_omni_asr


    # --- VLLM hooks ---

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        from audio_bench.model_src.vllm_backend import _input_audio_part
        return [
            {
                "role": "system",
                "content": (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                ),
            },
            {"role": "user", "content": [
                {"type": "text", "text": instruction + "\nPut the result in the following format: \\boxed{.}"},
                _input_audio_part(audio_array, sampling_rate),
            ]},
        ]

    def _postprocess_asr_text(self, text):
        return self._asr_text_processor(text)
