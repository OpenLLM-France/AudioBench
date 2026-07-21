import logging

from audio_bench.model_src.vllm_model import VLLMModel

logger = logging.getLogger(__name__)


class WhisperLargeV3(VLLMModel):

    name = "openai/whisper-large-v3"

    def __init__(self, gpu_memory_utilization=0.4, device=None):
        super().__init__(model_path="openai/whisper-large-v3", gpu_memory_utilization=gpu_memory_utilization, device=device)

    def load(self):
        from vllm import LLM, SamplingParams
        # Whisper's encoder-decoder context is 448 tokens, not the 4096 the generic loader
        # assumes, and it takes no gpu_memory_utilization override.
        self.llm = LLM(
            model=self.model_path,
            max_model_len=448,
            max_num_seqs=self.batch_size,
            limit_mm_per_prompt={"audio": 1},
        )
        self.sampling_params = SamplingParams(temperature=0, max_tokens=448)

    def generate(self, input):
        # Whisper is prompted through TextPrompt rather than the chat API, so it bypasses
        # _build_vllm_messages / the chunking path of the generic VLLMModel.generate().
        if not isinstance(input, list):
            input = [input]
        from vllm import TextPrompt
        from audio_bench.model_src.vllm_backend import _whisper_task_prompt

        all_prompts = [
            TextPrompt(
                prompt=_whisper_task_prompt(s['task_type'], s.get('language')),
                multi_modal_data={"audio": [(s["audio"]["array"], s["audio"]["sampling_rate"])]},
            )
            for s in input
        ]
        outputs = self.llm.generate(all_prompts, sampling_params=self.sampling_params, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outputs]
