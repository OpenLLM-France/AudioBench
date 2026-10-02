import logging
import os
import tempfile
from pathlib import Path


from audio_bench.model_src.vllm_model import VLLMModel

logger = logging.getLogger(__name__)


class AudioFlamingo(VLLMModel):

    name = "nvidia/audio-flamingo-3-hf"

    def __init__(self, model_path="nvidia/audio-flamingo-3-hf", gpu_memory_utilization=0.4, device=None):
        super().__init__(model_path=model_path, gpu_memory_utilization=gpu_memory_utilization, device=device)
        self.name = model_path


    # --- VLLM backend ---

    def load(self):
        from vllm import LLM, SamplingParams

        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        self.llm = LLM(
            model=self.model_path,
            max_model_len=4096,
            max_num_seqs=self.batch_size,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=self.gpu_memory_utilization,
            allowed_local_media_path=tempfile.gettempdir(),
        )
        self.sampling_params = SamplingParams(
            temperature=0, max_tokens=4096, repetition_penalty=1.2
        )

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        audio_path = self._write_temp_audio(audio_array, sampling_rate)
        audio_url = Path(audio_path).resolve().as_uri()
        return [
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "audio_url", "audio_url": {"url": audio_url}},
            ]},
        ]

    def _postprocess_asr_text(self, text):
        # Audio-Flamingo prefixes transcripts with descriptions ("The spoken content of the
        # audio is ...", "The audio contains ..."). Strip the preamble and unwrap any quotes.
        from audio_bench.asr_postprocess import strip_audio_description_preamble
        return strip_audio_description_preamble(text)
