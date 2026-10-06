import logging

from audio_bench.model_src.vllm_model import VLLMModel
from audio_bench.model_src.vllm_backend import _input_audio_part

logger = logging.getLogger(__name__)

# Granite Speech is an ASR/AST specialist trained on a few fixed English prompts, and its card
# states that punctuated non-English ASR and all punctuated AST "require" an English prompt. When
# True (default), ASR/AST use those native prompts instead of the dataset instruction (same spirit
# as Voxtral's transcription mode); set False to feed the dataset instruction for every task.
# Other tasks (QA, emotion, ...) always get the dataset instruction.
GRANITE_NATIVE_ASR_AST_PROMPTS = True
# The card's ASR prompt does not name the language, and on spontaneous / noisy French the model
# often translates to English instead ("Thank you very much."). When True, the dataset language
# is added to that prompt ("transcribe the speech in French ..."), which the card does not document.
# Off: it stops the English outputs but makes the model paraphrase like in AST ("époux" ->
# "mari", also in English), which costs 3-5 WER points on read speech (CommonVoice, Fleurs).
GRANITE_ASR_LANGUAGE_IN_PROMPT = False

_ASR_PROMPT = "transcribe the speech with proper punctuation and capitalization."
_ASR_PROMPT_WITH_LANGUAGE = "transcribe the speech in {language} with proper punctuation and capitalization."
_AST_PROMPT = "translate the speech to {language} with proper punctuation and capitalization."

# Language names as the model card spells them (AST targets; also used for ASR when
# GRANITE_ASR_LANGUAGE_IN_PROMPT is set).
_LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "ja": "Japanese", "it": "Italian", "zh": "Mandarin",
}


class GraniteSpeech(VLLMModel):
    """IBM Granite Speech (ibm-granite/granite-speech-4.1-2b by default) via vLLM.

    Supports ASR in EN/FR/DE/ES/PT/JA and AST to and from English (plus EN->IT/ZH). Pairs that
    do not involve English (e.g. ES->FR) are outside its training but still run with the same prompt.
    """

    name = "ibm-granite/granite-speech-4.1-2b"

    def __init__(self, model_path="ibm-granite/granite-speech-4.1-2b", gpu_memory_utilization=0.4, device=None):
        super().__init__(model_path=model_path, gpu_memory_utilization=gpu_memory_utilization, device=device)
        self.name = model_path

    def load(self):
        from vllm import SamplingParams

        super().load()
        # With the punctuated ASR prompt the model sometimes emits EOS right away (or after one
        # token) on perfectly normal speech: 24/500 empty outputs on French VoxPopuli. Forbidding
        # EOS for the first 2 tokens recovers full transcripts; greedy outputs longer than that
        # are unchanged.
        self.sampling_params = SamplingParams(temperature=0, max_tokens=512, min_tokens=2)

    def _generate_vllm(self, inputs):
        if GRANITE_NATIVE_ASR_AST_PROMPTS:
            inputs = [{**inp, "instruction": self._native_instruction(inp)} for inp in inputs]
        return super()._generate_vllm(inputs)

    def _native_instruction(self, inp):
        task_type = inp.get("task_type")
        if task_type == "ASR":
            language = _LANGUAGE_NAMES.get((inp.get("language") or "").lower())
            if GRANITE_ASR_LANGUAGE_IN_PROMPT and language:
                return _ASR_PROMPT_WITH_LANGUAGE.format(language=language)
            return _ASR_PROMPT
        if task_type == "AST":
            # AST language is the pair "<src>-<tgt>", e.g. "FR-EN".
            target = (inp.get("language") or "").lower().split("-")[-1]
            if target in _LANGUAGE_NAMES:
                return _AST_PROMPT.format(language=_LANGUAGE_NAMES[target])
            logger.warning(f"Unknown AST target language {inp.get('language')!r}: using the dataset instruction.")
        return inp["instruction"]

    # --- VLLM hooks ---

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        # vLLM inserts the model's <|audio|> placeholder for the audio part.
        return [
            {"role": "user", "content": [
                _input_audio_part(audio_array, sampling_rate),
                {"type": "text", "text": instruction},
            ]},
        ]

    def _postprocess_asr_text(self, text):
        return text.strip()
