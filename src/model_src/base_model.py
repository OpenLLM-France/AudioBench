import os
import tempfile

import numpy as np
import soundfile as sf
import torch
import logging

logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all AudioBench models."""

    supports_vllm = False
    max_audio_duration = 30

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_name = None
        self.model_name = None
        self.backend = None
        self._temp_files = []

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
            try:
                os.remove(path)
            except OSError:
                pass
        self._temp_files.clear()

    # --- Audio preprocessing ---

    def _prepare_audio_segments(self, audio_array, sampling_rate, task_type):
        """Chunk / truncate / pad audio based on max_audio_duration.

        Returns (segments, mode) where mode is 'chunked', 'truncated', or 'normal'.
        """
        audio_duration = len(audio_array) / sampling_rate
        max_dur = self.max_audio_duration

        if audio_duration > max_dur and task_type == 'ASR':
            logger.info(f'Audio duration is more than {max_dur} seconds. Chunking and inferring separately.')
            chunks = []
            for i in range(0, len(audio_array), max_dur * sampling_rate):
                chunks.append(audio_array[i:i + max_dur * sampling_rate])
            return chunks, 'chunked'

        if audio_duration > max_dur:
            logger.info(f'Audio duration is more than {max_dur} seconds. Taking first {max_dur} seconds.')
            return [audio_array[:max_dur * sampling_rate]], 'truncated'

        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        return [audio_array], 'normal'

    # --- Public API (called by main_evaluate) ---

    def generate(self, input):
        try:
            if self.backend == "vllm":
                if not isinstance(input, list):
                    input = [input]
                return self.generate_vllm(input)
            with torch.no_grad():
                return self._generate(input)
        finally:
            self._cleanup_temp_files()

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
            max_num_seqs=5,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=0.6,
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
            audio_array = inp["audio"]["array"]
            sampling_rate = inp["audio"]["sampling_rate"]
            instruction = inp["instruction"]
            is_asr = inp['task_type'] == 'ASR'

            segments, mode = self._prepare_audio_segments(audio_array, sampling_rate, inp['task_type'])

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

        outputs = self.llm.chat(all_messages, sampling_params=self.sampling_params, **self._vllm_chat_kwargs())

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
