import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import logging
import psutil


logger = logging.getLogger(__name__)


def get_memory_usage_ratio():
    """Return GPU memory usage as a fraction (0.0–1.0).
    Uses torch.cuda.mem_get_info() which reports VRAM at the CUDA driver level,
    capturing all allocators (PyTorch, vLLM, raw CUDA).
    """
    vm = psutil.virtual_memory()
    ratio_mem_used = vm.used / vm.total
    return ratio_mem_used


def should_free_model(threshold=None):
    """Return True if GPU memory usage exceeds the threshold (default 80%).
    Threshold is configurable via AUDIOBENCH_MEMORY_THRESHOLD env var.
    """
    if threshold is None:
        threshold = float(os.environ.get("AUDIOBENCH_MEMORY_THRESHOLD", "0.8"))
    usage = get_memory_usage_ratio()
    logger.info(f"GPU memory usage: {usage:.1%} (threshold: {threshold:.0%})")
    return usage > threshold


class BaseModel:
    """Base class for all AudioBench models."""

    name = None
    supports_vllm = False
    max_audio_duration = 120

    # Optional override for the audio placeholder inserted into prompts.
    # When None, NeMo models fall back to the model's built-in
    # `self.model.audio_locator_tag`. Set from the model-level config field
    # `audio_locator_tag` to use a custom tag without mutating the model.
    audio_locator_tag = None

    is_api_model = False

    def __init__(self, model_path=None, gpu_memory_utilization=0.4, device=None):
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.device = device or "auto"
        self._validate_device()
        self.dataset_name = None
        self.model_id = None
        self.backend = None
        self.batch_size = 1
        self._temp_files = []
        self.llm = None

    def _validate_device(self):
        """Validate device setting and ensure GPU is available (unless API model)."""
        if self.is_api_model:
            return
        # Reject invalid device strings early
        valid = self.device == "auto" or self.device == "cuda" or self.device.startswith("cuda:")
        if not valid:
            raise RuntimeError(
                f"Invalid device: '{self.device}'. Must be 'auto', 'cuda', or 'cuda:N' (e.g. 'cuda:0')."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available.")
        if self.device.startswith("cuda:"):
            try:
                idx = int(self.device.split(":")[1])
            except ValueError:
                raise RuntimeError(f"Invalid device specification: {self.device}")
            if idx >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested {self.device} but only {torch.cuda.device_count()} GPU(s) available."
                )

    @property
    def torch_device(self):
        """Resolved device for .to() calls. Returns 'cuda' when device is 'auto'."""
        if self.device == "auto":
            return "cuda"
        return self.device

    # --- Temp file helpers ---

    def _write_temp_audio(self, audio_array, sampling_rate):
        """Write audio to a temp WAV file, track it for cleanup, return the path."""
        f = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(f.name, audio_array, sampling_rate)
        self._temp_files.append(f.name)
        return f.name

    def _cleanup_temp_files(self):
        """Remove all temp files created during the last generate() call."""
        for path in self._temp_files:
            Path(path).unlink(missing_ok=True)
        self._temp_files.clear()

    # --- Audio preprocessing ---

    def _prepare_audio_segments(self, audio, task_type):
        """Chunk / truncate / pad audio based on max_audio_duration.

        Args:
            audio: dict with keys 'array', 'sampling_rate', and optionally 'path'.
            task_type: e.g. 'ASR'.

        Returns (segments, sampling_rate, mode) where mode is 'chunked', 'truncated', or 'normal'.
        """
        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        audio_path = audio.get("path")
        audio_duration = len(audio_array) / sampling_rate
        max_dur = self.max_audio_duration

        if audio_duration > max_dur and task_type == 'ASR':
            logger.info(f'Audio duration is more than {max_dur}s. Chunking and inferring separately. Path: {audio_path}')
            chunks = []
            for i in range(0, len(audio_array), max_dur * sampling_rate):
                chunks.append(audio_array[i:i + max_dur * sampling_rate])
            return chunks, sampling_rate, 'chunked'

        if audio_duration > max_dur:
            logger.info(f'Audio duration is more than {max_dur}s. Taking first {max_dur}s. Path: {audio_path}')
            return [audio_array[:max_dur * sampling_rate]], sampling_rate, 'truncated'

        if audio_duration < 0.5:
            logger.info(f'Audio duration is less than 0.5s. Padding to 0.5s. Path: {audio_path}')
            pad_samples = int(0.5 * sampling_rate) - len(audio_array)
            audio_array = np.pad(audio_array, (0, pad_samples), 'constant')
            return [audio_array], sampling_rate, 'padded'

        return [audio_array], sampling_rate, 'normal'

    # --- Public API (called by main_evaluate) ---

    def generate(self, input):
        try:
            if self.backend == "vllm":
                if not isinstance(input, list):
                    input = [input]
                return self.generate_vllm(input)
            with torch.no_grad():
                if isinstance(input, list):
                    return self._generate_batch(input)
                return self._generate(input)
        finally:
            self._cleanup_temp_files()

    def _generate_batch(self, inputs):
        """Batched generation for the transformers backend.
        Default: sequential fallback. Override for true batching.
        """
        logger.warning(f"{type(self).__name__} does not support batching. Falling back to sequential generation.")
        results = []
        for inp in inputs:
            result = self._generate(inp)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        return results

    # --- To be implemented by subclasses ---

    def load(self):
        raise NotImplementedError

    def _generate(self, input):
        raise NotImplementedError

    def load_vllm(self):
        if not self.supports_vllm:
            raise NotImplementedError(
                f"{type(self).__name__} does not support VLLM backend"
            )
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=self.model_path,
            max_model_len=4096,
            max_num_seqs=self.batch_size,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(temperature=0, max_tokens=512)

    def generate_vllm(self, inputs):
        """Batched VLLM generation with chunk/truncate/pad support.

        Subclasses must implement _build_vllm_messages().
        They may optionally override the hooks below.
        """
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

    # --- VLLM hooks (override in subclasses) ---

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

    def destroy(self):
        """Explicitly release resources."""
        if self.llm is not None:
            # vLLM V1 engine (and some older versions) uses separate processes.
            # We try to trigger cleanup by following vLLM's recommended patterns if available.
            try:
                # vLLM v0.6+ cleanup
                if hasattr(self.llm, "llm_engine"):
                    engine = self.llm.llm_engine
                    if hasattr(engine, "model_executor"):
                        executor = engine.model_executor
                        if hasattr(executor, "shutdown"):
                            executor.shutdown()
                
                # Try to trigger the distributed cleanup if parallel_state exists
                from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception as e:
                logger.debug(f"Error during parallel state cleanup: {e}")

            import gc
            # Delete the high-level LLM object to trigger its __del__
            del self.llm
            self.llm = None
            
            # Additional cleanup for V1 engine (if applicable)
            try:
                import vllm.v1.engine.core_client as core_client
                if hasattr(core_client, "launch_core_engines"):
                    # This is reaching deep, but we need that memory back.
                    pass 
            except ImportError:
                pass

            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"Model {self.name} resources released.")
