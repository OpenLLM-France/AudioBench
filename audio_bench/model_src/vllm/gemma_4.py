import logging

from audio_bench.model_src.vllm_model import VLLMModel
from audio_bench.model_src.vllm_backend import _input_audio_part

logger = logging.getLogger(__name__)

# The model card gives dedicated ASR / AST prompts. When True (default), ASR/AST use them instead
# of the dataset instruction (same spirit as Voxtral's transcription mode); set False to feed the
# dataset instruction for every task. Other tasks (QA, emotion, ...) always get the dataset one.
GEMMA_NATIVE_ASR_AST_PROMPTS = True

# Verbatim from the model card ("6. Audio").
_ASR_PROMPT = (
    "Transcribe the following speech segment in {language} into {language} text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, "
    "and write 3 instead of three."
)
_AST_PROMPT = (
    "Transcribe the following speech segment in {source}, then translate it into {target}.\n"
    "When formatting the answer, first output the transcription in {source}, then one newline, "
    "then output the string '{target}: ', then the translation in {target}."
)

_LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
}


def _patch_vllm_gemma4_audio_batching():
    """Work around a vLLM (0.27.1) bug: batching audios of different lengths crashes Gemma 4.

    The processor unpads each audio's features and relies on the batched() field config to re-pad
    them, but they reach _process_audio_input as a list and `.squeeze(1)` fails
    ("'list' object has no attribute 'squeeze'"). Re-pad the lists here (zeros / mask False).

    The patch lives in this process, so the engine core must run in-process too
    (VLLM_ENABLE_V1_MULTIPROCESSING=0).
    """
    import os
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from vllm.model_executor.models import gemma4_mm

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    cls = gemma4_mm.Gemma4ForConditionalGeneration
    if getattr(cls, "_audiobench_padded", False):
        return
    original = cls._process_audio_input

    def _pad(items, dim):
        # Each item is [T, D] (features) or [T] (mask), possibly with extra leading size-1 dims.
        items = [x.reshape(x.shape[-dim:]) for x in items]
        return pad_sequence(items, batch_first=True).unsqueeze(1)

    def _process_audio_input(self, audio_input):
        if isinstance(audio_input["input_features_padded"], (list, tuple)):
            audio_input = {
                "input_features_padded": _pad(audio_input["input_features_padded"], 2),
                "input_features_mask": _pad(
                    [m.to(torch.bool) for m in audio_input["input_features_mask"]], 1),
            }
        return original(self, audio_input)

    cls._process_audio_input = _process_audio_input
    cls._audiobench_padded = True


class Gemma4(VLLMModel):
    """Google Gemma 4 with audio input (google/gemma-4-E2B-it by default, E4B-it also works) via vLLM.

    Only the E2B / E4B checkpoints carry the audio encoder that vLLM's
    Gemma4ForConditionalGeneration supports (the 12B is the encoder-free "Unified" variant).

    The model card caps audio at 30 seconds, so longer ASR audio is chunked and other tasks are
    truncated (max_audio_duration). It also recommends putting the audio AFTER the text.
    Thinking stays off: it is only triggered by a <|think|> token in the system prompt.
    """

    name = "google/gemma-4-E2B-it"
    max_audio_duration = 30

    def __init__(self, model_path="google/gemma-4-E2B-it", gpu_memory_utilization=0.4, device=None):
        super().__init__(model_path=model_path, gpu_memory_utilization=gpu_memory_utilization, device=device)
        self.name = model_path

    # --- vLLM engine ---

    def load(self):
        from vllm import LLM, SamplingParams

        _patch_vllm_gemma4_audio_batching()
        self.llm = LLM(
            model=self.model_path,
            max_model_len=4096,
            max_num_seqs=self.batch_size,
            # No image/video in this benchmark: skip their memory profiling.
            limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(temperature=0, max_tokens=512)

    def _generate_vllm(self, inputs):
        if not GEMMA_NATIVE_ASR_AST_PROMPTS:
            return super()._generate_vllm(inputs)
        inputs = [{**inp, "instruction": self._native_instruction(inp)} for inp in inputs]
        results = super()._generate_vllm(inputs)
        # The AST prompt makes the model output "<transcription>\n<Target>: <translation>":
        # keep only the translation.
        for i, inp in enumerate(inputs):
            target = self._ast_languages(inp)
            if target and results[i] is not None:
                results[i] = self._extract_translation(results[i], target[1])
        return results

    def _ast_languages(self, inp):
        """(source, target) language names for an AST input, or None."""
        if inp.get("task_type") != "AST":
            return None
        # AST language is the pair "<src>-<tgt>", e.g. "FR-EN".
        codes = (inp.get("language") or "").lower().split("-")
        if len(codes) != 2 or not all(c in _LANGUAGE_NAMES for c in codes):
            return None
        return _LANGUAGE_NAMES[codes[0]], _LANGUAGE_NAMES[codes[1]]

    def _native_instruction(self, inp):
        task_type = inp.get("task_type")
        if task_type == "ASR":
            lang = _LANGUAGE_NAMES.get((inp.get("language") or "").lower())
            if lang:
                return _ASR_PROMPT.format(language=lang)
        elif task_type == "AST":
            langs = self._ast_languages(inp)
            if langs:
                return _AST_PROMPT.format(source=langs[0], target=langs[1])
        else:
            return inp["instruction"]
        logger.warning(f"Unknown {task_type} language {inp.get('language')!r}: using the dataset instruction.")
        return inp["instruction"]

    @staticmethod
    def _extract_translation(text, target):
        marker = f"{target}:"
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
        # Marker missing: drop the transcription line if there is one.
        lines = [l for l in text.strip().split("\n") if l.strip()]
        return lines[-1].strip() if lines else text.strip()

    # --- VLLM hooks ---

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        return [
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                _input_audio_part(audio_array, sampling_rate),
            ]},
        ]

    def _postprocess_asr_text(self, text):
        # Chat model: recover the transcript if it is wrapped in a preamble / quotes.
        from audio_bench.asr_postprocess import postprocess_asr_prediction
        return postprocess_asr_prediction(text)

    def _rewrite_instruction(self, instruction, task_type):
        # Neutralize "file" framing in ASR/AST prompts, which makes chat models refuse.
        if task_type in ("ASR", "AST"):
            from audio_bench.model_src.asr_instruction import neutralize_file_references
            return neutralize_file_references(instruction)
        return instruction
