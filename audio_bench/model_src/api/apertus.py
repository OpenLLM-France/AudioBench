import logging
import os
from concurrent.futures import ThreadPoolExecutor

from audio_bench.model_src.base_model import BaseModel
from audio_bench.model_src.vllm_backend import _audio_to_base64_wav

logger = logging.getLogger(__name__)

# Apertus 1.5 only runs on the swiss-ai forks of vLLM and transformers, absent from the
# AudioBench images: it is served apart (tools/serve_apertus_dgx.sh) and queried over the
# OpenAI-compatible API. The server resamples the audio to the model's 24 kHz.
APERTUS_BASE_URL = os.environ.get("APERTUS_BASE_URL", "http://172.17.0.1:8120/v1")
APERTUS_CONCURRENCY = int(os.environ.get("APERTUS_CONCURRENCY", "32"))

# The model card has no ASR prompt: one of the dataset prompts with the lowest French WER,
# which needs no language name.
_ASR_PROMPT = "Transcribe what is said in the source language"


class Apertus(BaseModel):
    """swiss-ai/Apertus-v1.5-8B through a `vllm serve` endpoint, thinking off (its default)."""

    served_model = "swiss-ai/Apertus-v1.5-8B"
    native_backend = "vllm-server"
    is_api_model = True  # no local GPU: the server owns it

    def __init__(self, device=None):
        super().__init__(device=device)
        self.name = self.served_model

    def load(self):
        from openai import OpenAI

        self.client = OpenAI(base_url=APERTUS_BASE_URL, api_key="EMPTY", timeout=1800)
        served = [m.id for m in self.client.models.list().data]
        if self.served_model not in served:
            raise RuntimeError(f"{self.served_model} is not served at {APERTUS_BASE_URL} (served: {served})")
        logger.info(f"Using {self.served_model} served at {APERTUS_BASE_URL}")

    def prediction_chunk_size(self, batch_size):
        return max(100, APERTUS_CONCURRENCY * 4)

    def _generate(self, input):
        return self._generate_batch([input])[0]

    def _generate_batch(self, inputs):
        jobs, meta = [], []
        for i, inp in enumerate(inputs):
            segments, sampling_rate, _ = self._prepare_audio_segments(inp["audio"], inp["task_type"])
            instruction = _ASR_PROMPT if inp["task_type"] == "ASR" else inp["instruction"]
            for seg in segments:
                jobs.append((seg, sampling_rate, instruction))
                meta.append(i)

        with ThreadPoolExecutor(APERTUS_CONCURRENCY) as pool:
            outputs = list(pool.map(lambda job: self._request(*job), jobs))

        # Chunked ASR audio: the chunk transcripts are joined in order.
        results = [[] for _ in inputs]
        for i, text in zip(meta, outputs):
            results[i].append(text)
        return [" ".join(parts) for parts in results]

    def _request(self, audio_array, sampling_rate, instruction):
        audio_b64 = _audio_to_base64_wav(audio_array, sampling_rate)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
        ]}]
        completion = self.client.chat.completions.create(
            model=self.served_model,
            messages=messages,
            temperature=0,
            max_tokens=512,
            seed=0,
        )
        return (completion.choices[0].message.content or "").strip()
