import logging

from audio_bench.model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


class VLLMModel(BaseModel):
    """Base for models that generate through vllm.

    Subclassing IS the declaration: a model either runs on vllm or it does not, so there is
    no `supports_vllm` flag and no backend branch anywhere -- the caller just does
    `model.load()` / `model.generate()` and polymorphism picks the implementation.

    Subclasses must implement _build_vllm_messages(); the other hooks are optional.
    """

    native_backend = "vllm"

    # vllm builds all chat messages up front, so it is fed in larger slices than the
    # per-model batch_size (which caps concurrency via max_num_seqs, not memory here).
    # main_evaluate asks the model for this rather than testing the backend name.
    def prediction_chunk_size(self, batch_size):
        return max(100, batch_size * 4)

    def load(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=self.model_path,
            max_model_len=4096,
            max_num_seqs=self.batch_size,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(temperature=0, max_tokens=512)

    def generate(self, input):
        """Batched vllm generation with chunk/truncate/pad support."""
        if not isinstance(input, list):
            input = [input]
        try:
            return self._generate_vllm(input)
        finally:
            self._cleanup_temp_files()

    def _generate_vllm(self, inputs):
        results = [None] * len(inputs)
        all_messages = []
        meta = []
        chunk_buffers = {}

        for i, inp in enumerate(inputs):
            instruction = inp["instruction"]
            is_asr = inp['task_type'] == 'ASR'

            segments, sampling_rate, mode = self._prepare_audio_segments(inp["audio"], inp['task_type'])

            if mode == 'chunked':
                chunk_buffers[i] = []
                for seg in segments:
                    seg, sr = self._preprocess_audio_for_vllm(seg, sampling_rate)
                    all_messages.append(self._build_vllm_messages(seg, sr, instruction))
                    meta.append(('chunk', i))
            else:
                seg = segments[0]
                seg, sr = self._preprocess_audio_for_vllm(seg, sampling_rate)
                all_messages.append(self._build_vllm_messages(seg, sr, instruction))
                meta.append(('normal', i, is_asr))

        outputs = self.llm.chat(all_messages, sampling_params=self.sampling_params, use_tqdm=False, **self._vllm_chat_kwargs())

        for output, m in zip(outputs, meta):
            text = output.outputs[0].text
            if m[0] == 'chunk':
                chunk_buffers[m[1]].append(self._postprocess_asr_text(text))
            else:
                _, result_idx, is_asr = m
                if is_asr:
                    text = self._postprocess_asr_text(text)
                results[result_idx] = text

        for result_idx, chunks in chunk_buffers.items():
            results[result_idx] = ' '.join(chunks)

        return results

    # --- Hooks (override in subclasses) ---

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        """Build the chat messages list for a single audio. Must be overridden."""
        raise NotImplementedError

    def _preprocess_audio_for_vllm(self, audio_array, sampling_rate):
        """Optional audio preprocessing (e.g. resampling). Default: identity."""
        return audio_array, sampling_rate

    def _postprocess_asr_text(self, text):
        """Optional ASR text postprocessing. Default: identity."""
        return text

    def _vllm_chat_kwargs(self):
        """Extra kwargs passed to llm.chat(). Default: empty dict."""
        return {}
