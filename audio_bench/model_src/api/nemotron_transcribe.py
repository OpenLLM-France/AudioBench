import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from audio_bench.model_src.base_model import BaseModel
from audio_bench.model_src.vllm_backend import _audio_to_base64_wav

logger = logging.getLogger(__name__)

# NVIDIA Nemotron-3.5-Transcribe is a NeMo SALM whose vllm plugin (NeMo Speech main) needs
# vLLM >= 0.31: it cannot load in the AudioBench image, so it is served apart
# (tools/serve_nemotron_transcribe_dgx.sh) and queried over the OpenAI-compatible API.
NEMOTRON_BASE_URL = os.environ.get("NEMOTRON_BASE_URL", "http://172.17.0.1:8119/v1")
NEMOTRON_CONCURRENCY = int(os.environ.get("NEMOTRON_CONCURRENCY", "32"))

# Model card settings: reasoning off at temperature 0 for transcription, translation and
# language ID; reasoning on (temperature 0.6, top-p 0.95) for audio QA and the other
# understanding tasks. The no-thinking variant runs every task without reasoning.
_NO_THINKING_TASKS = {"ASR", "AST", "SPOKEN LANGUAGE IDENTIFICATION"}

# Model card's verbatim transcription prompt, with the language named (as in asr_benchmark).
_ASR_PROMPT = ("Audio transcription. Language: {language}. Wording: verbatim, including fillers and "
               "repetitions. Formatting: punctuation and capitalization. Output: transcript only.")
_AST_PROMPT = "Please translate the given speech to {language}."
_LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "nl": "Dutch", "pt": "Portuguese", "ar": "Arabic",
}

_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)


class NemotronTranscribe(BaseModel):
    """nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B through a `vllm serve` endpoint.

    ASR and AST use the model card's native prompts; the other tasks get the dataset
    instruction. With thinking=False no task uses reasoning (a second leaderboard row).
    """

    served_model = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
    native_backend = "vllm-server"
    is_api_model = True  # no local GPU: the server owns it

    def __init__(self, thinking=True, device=None):
        super().__init__(device=device)
        self.thinking = thinking
        self.name = f"{self.served_model} ({'thinking' if thinking else 'no thinking'})"

    def load(self):
        from openai import OpenAI

        self.client = OpenAI(base_url=NEMOTRON_BASE_URL, api_key="EMPTY", timeout=1800)
        served = [m.id for m in self.client.models.list().data]
        if self.served_model not in served:
            raise RuntimeError(f"{self.served_model} is not served at {NEMOTRON_BASE_URL} (served: {served})")
        logger.info(f"Using {self.served_model} served at {NEMOTRON_BASE_URL} (thinking: {self.thinking})")

    def prediction_chunk_size(self, batch_size):
        return max(100, NEMOTRON_CONCURRENCY * 4)

    def _generate(self, input):
        return self._generate_batch([input])[0]

    def _generate_batch(self, inputs):
        jobs, meta = [], []
        for i, inp in enumerate(inputs):
            segments, sampling_rate, mode = self._prepare_audio_segments(inp["audio"], inp["task_type"])
            instruction = self._native_instruction(inp)
            for seg in segments:
                jobs.append((seg, sampling_rate, instruction, inp["task_type"]))
                meta.append(i)

        with ThreadPoolExecutor(NEMOTRON_CONCURRENCY) as pool:
            outputs = list(pool.map(lambda job: self._request(*job), jobs))

        # Chunked ASR audio: the chunk transcripts are joined in order.
        results = [[] for _ in inputs]
        for i, text in zip(meta, outputs):
            results[i].append(text)
        return [" ".join(parts) for parts in results]

    def _native_instruction(self, inp):
        task_type = inp["task_type"]
        language = (inp.get("language") or "").lower()
        if task_type == "ASR" and language in _LANGUAGE_NAMES:
            return _ASR_PROMPT.format(language=_LANGUAGE_NAMES[language])
        if task_type == "AST":
            # AST language is the pair "<src>-<tgt>", e.g. "FR-EN".
            target = language.split("-")[-1]
            if target in _LANGUAGE_NAMES:
                return _AST_PROMPT.format(language=_LANGUAGE_NAMES[target])
        if task_type in ("ASR", "AST"):
            logger.warning(f"Unknown {task_type} language {inp.get('language')!r}: using the dataset instruction.")
        return inp["instruction"]

    def _request(self, audio_array, sampling_rate, instruction, task_type):
        thinking = self.thinking and task_type not in _NO_THINKING_TASKS
        if thinking:
            sampling = dict(temperature=0.6, top_p=0.95, max_tokens=20480)
        else:
            sampling = dict(temperature=0, top_p=1.0, max_tokens=4096)
        audio_b64 = _audio_to_base64_wav(audio_array, sampling_rate)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
        ]}]
        completion = self.client.chat.completions.create(
            model=self.served_model,
            messages=messages,
            seed=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
            **sampling,
        )
        text = completion.choices[0].message.content or ""
        # Served without a reasoning parser, the reasoning stays in the content.
        return _THINK_RE.sub("", text).strip()
