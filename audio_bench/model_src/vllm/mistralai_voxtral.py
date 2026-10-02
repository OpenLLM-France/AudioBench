import io
import logging

import soundfile as sf

from audio_bench.model_src.vllm_model import VLLMModel
from audio_bench.model_src.vllm_backend import _input_audio_part

logger = logging.getLogger(__name__)

# Voxtral exposes a DEDICATED transcription mode (mistral_common TranscriptionRequest) -- the path
# its model card means by "excels at speech transcription". When True (default), ASR runs through
# that native mode (implementation "B"); set False to use the generic chat path for ASR too
# (implementation "A": a plain transcription instruction). BOTH drop the old "\\boxed{.}" suffix,
# which is the LaTeX math-answer convention and pushed Voxtral into reasoning/hallucination on ASR
# (e.g. "1. **Step 1:** ... \\frac{1}{2} ...") -- the cause of its catastrophic WER here.
VOXTRAL_ASR_TRANSCRIPTION_MODE = True

# Languages Voxtral's transcription mode supports; anything else falls back to the chat path (A).
_VOXTRAL_TRANSCRIPTION_LANGS = {"en", "fr", "de", "es", "it", "pt", "nl", "hi"}


class Voxtral(VLLMModel):
    """Mistral Voxtral audio LLMs via vLLM. Same interface for the whole family -- the
    3B (Voxtral-Mini-3B-2507, default) and the 24B (Voxtral-Small-24B-2507) share the
    VoxtralForConditionalGeneration architecture, the mistral tokenizer mode, and the native
    transcription protocol; only the checkpoint (and its VRAM footprint) differ.
    """

    name = "mistralai/Voxtral-Mini-3B-2507"

    def __init__(self, model_path="mistralai/Voxtral-Mini-3B-2507", gpu_memory_utilization=0.4, device=None):
        super().__init__(model_path=model_path, gpu_memory_utilization=gpu_memory_utilization, device=device)
        self.name = model_path  # display name tracks the actual checkpoint (Mini-3B vs Small-24B)
        self._mistral_tokenizer = None

    # --- vLLM engine ---

    def load(self):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=self.model_path,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            max_model_len=4096,
            max_num_seqs=self.batch_size,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(temperature=0, max_tokens=512)

    # --- Implementation A: generic chat (AST, all non-ASR tasks, and ASR when B is disabled) ---

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        # Pass the dataset instruction through unchanged. NO "\\boxed{.}" suffix: it is a math
        # convention that made Voxtral hallucinate reasoning instead of transcribing/translating.
        return [
            {"role": "user", "content": [
                _input_audio_part(audio_array, sampling_rate),
                {"type": "text", "text": instruction},
            ]},
        ]

    def _postprocess_asr_text(self, text):
        return text.strip()

    # --- Implementation B: Voxtral native transcription mode for ASR ---

    def _generate_vllm(self, inputs):
        """Route ASR to Voxtral's transcription mode (B); everything else uses the chat path (A).

        When VOXTRAL_ASR_TRANSCRIPTION_MODE is False, ASR also goes through the chat path (A).
        """
        if not VOXTRAL_ASR_TRANSCRIPTION_MODE:
            return super()._generate_vllm(inputs)

        results = [None] * len(inputs)
        chat_idx = [i for i, inp in enumerate(inputs) if not self._use_transcription(inp)]
        tr_idx = [i for i, inp in enumerate(inputs) if self._use_transcription(inp)]

        if chat_idx:
            for i, r in zip(chat_idx, super()._generate_vllm([inputs[i] for i in chat_idx])):
                results[i] = r
        if tr_idx:
            for i, r in zip(tr_idx, self._transcribe([inputs[i] for i in tr_idx])):
                results[i] = r
        return results

    def _use_transcription(self, inp):
        """True if this input should go through Voxtral's native transcription mode."""
        if inp.get("task_type") != "ASR":
            return False
        lang = (inp.get("language") or "").lower()
        return lang in _VOXTRAL_TRANSCRIPTION_LANGS  # unsupported language -> chat fallback (A)

    def _mistral_tok(self):
        if self._mistral_tokenizer is None:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            self._mistral_tokenizer = MistralTokenizer.from_hf_hub(self.model_path)
        return self._mistral_tokenizer

    def _transcribe(self, inputs):
        """Transcribe a batch of ASR inputs via mistral_common's TranscriptionRequest.

        Builds the native transcription prompt (audio + language hint) for each sample and runs it
        through vLLM's offline engine. ASR audio is capped upstream (config max_audio_duration), so
        a single segment per sample is enough -- no chunking here.
        """
        from mistral_common.audio import Audio
        from mistral_common.protocol.transcription.request import TranscriptionRequest
        try:
            from mistral_common.protocol.transcription.request import RawAudio
        except ImportError:  # moved in mistral_common >= 1.10
            from mistral_common.protocol.instruct.chunk import RawAudio

        tok = self._mistral_tok()
        prompts = []
        for inp in inputs:
            segments, sr, _mode = self._prepare_audio_segments(inp["audio"], "ASR")
            arr, sr = self._preprocess_audio_for_vllm(segments[0], sr)
            buf = io.BytesIO()
            sf.write(buf, arr, sr, format="WAV")
            buf.seek(0)
            # TranscriptionRequest.audio wants a RawAudio (not an Audio); from_audio wraps it.
            audio = RawAudio.from_audio(Audio.from_bytes(buf.read()))
            lang = (inp.get("language") or "en").lower()
            enc = tok.encode_transcription(
                TranscriptionRequest(model=self.model_path, audio=audio, language=lang)
            )
            prompts.append({
                "prompt_token_ids": enc.tokens,
                "multi_modal_data": {"audio": [(arr, sr)]},
            })

        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outputs]
