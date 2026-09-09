import logging

from audio_bench.model_src.vllm_model import VLLMModel
from audio_bench.model_src.vllm_backend import _input_audio_part

logger = logging.getLogger(__name__)


class Qwen3Omni(VLLMModel):
    """Qwen3-Omni (e.g. Qwen/Qwen3-Omni-30B-A3B-Instruct) via vLLM.

    Text-out only: vLLM's LLM path drives the *thinker*, so predictions come back as
    text (transcriptions / answers) with no speech synthesis. The Instruct variant
    answers directly (it does not emit <think> reasoning), so no thinking-block
    stripping is needed.

    Unlike the in-repo Qwen2.5-Omni class, this does NOT force a \\boxed{} answer
    format: it passes the dataset instruction through unchanged and only strips the
    raw output, matching the reference ASR_Benchmark integration.

    This is a 30B-A3B MoE (~60 GB in bf16): it needs a lot of VRAM. It is loaded with
    a high gpu_memory_utilization and sharded with tensor parallelism across every
    visible GPU (governed by CUDA_VISIBLE_DEVICES) so it fits on a typical multi-GPU
    node without extra flags.
    """

    name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    def __init__(self, model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                 gpu_memory_utilization=0.4, device=None):
        super().__init__(model_path=model_path,
                         gpu_memory_utilization=gpu_memory_utilization, device=device)

    # --- vLLM engine ---

    def load(self):
        import torch
        from vllm import LLM, SamplingParams

        # A 30B MoE cannot load at the generic 0.4 factory default (main_evaluate does
        # not forward gpu_memory_utilization at all), so raise a safety floor. An
        # explicit value from the config is respected as long as it clears the floor.
        # The floor is deliberately conservative (0.6) for unified-memory boxes like
        # the DGX Spark, where this fraction is carved out of the SAME pool the OS and
        # CPU-side audio buffers use -- 0.9 there would starve the host and OOM.
        gpu_mem = max(self.gpu_memory_utilization, 0.6)
        if gpu_mem != self.gpu_memory_utilization:
            logger.info(
                f"Raising gpu_memory_utilization {self.gpu_memory_utilization} -> {gpu_mem} "
                "(Qwen3-Omni-30B needs a large, contiguous slice of memory)."
            )
        # Shard across every visible GPU (respects CUDA_VISIBLE_DEVICES). A single
        # GPU is used as-is; the model likely needs 2+ 80GB GPUs to fit.
        tp = max(1, torch.cuda.device_count())
        logger.info(
            f"Loading {self.model_path} with tensor_parallel_size={tp}, "
            f"gpu_memory_utilization={gpu_mem}."
        )

        self.llm = LLM(
            model=self.model_path,
            max_model_len=8192,
            max_num_seqs=self.batch_size,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=gpu_mem,
            tensor_parallel_size=tp,
        )
        self.sampling_params = SamplingParams(temperature=0, max_tokens=512)

    # --- VLLM hooks ---

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        # No system message: vLLM applies the model's chat-template default, matching
        # the reference integration (which let apply_chat_template inject it).
        return [
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                _input_audio_part(audio_array, sampling_rate),
            ]},
        ]

    def _postprocess_asr_text(self, text):
        return text.strip()
