import re
import logging

import numpy as np
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor

from audio_bench.model_src.base_model import BaseModel
from audio_bench.model_src.vllm_backend import _input_audio_part

logger = logging.getLogger(__name__)


def _post_process_voxtral_asr(model_output):
    # Try \boxed{\text{...}}
    m = re.search(r"\\boxed\{\\text\{(.*?)\}\}", model_output, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # Try \boxed{...} (non-empty) — take last occurrence
    matches = list(re.finditer(r"\\boxed\{(.+?)\}", model_output))
    if matches:
        content = matches[-1].group(1).strip()
        # Clean up escaped braces \{...\}
        if content.startswith("\\{"):
            content = content[2:]
        if content.endswith("\\"):
            content = content[:-1]
        content = content.strip()
        if content:
            return content

    # \boxed{} empty → text is before it
    if r"\boxed{}" in model_output:
        text = model_output.split(r"\boxed{}")[0].strip()
        if text:
            return text

    return model_output


class Voxtral(BaseModel):

    supports_vllm = True

    def __init__(self, model_path="mistralai/Voxtral-Mini-3B-2507", gpu_memory_utilization=0.4, device=None):
        super().__init__(model_path=model_path, gpu_memory_utilization=gpu_memory_utilization, device=device)

    # --- Transformers backend ---

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = VoxtralForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, device_map=self.device).eval()
        logger.info(f"Model loaded: {self.model_path}")

    def _generate(self, input):

        audio_array    = input["audio"]["array"]
        sampling_rate  = input["audio"]["sampling_rate"]
        audio_duration = len(audio_array) / sampling_rate

        if audio_duration < 0.5:
            logger.info('Audio duration is less than 0.5 second. Padding the audio to 0.5 second.')
            pad_samples = int(0.5 * sampling_rate) - len(audio_array)
            audio_array = np.pad(audio_array, (0, pad_samples), 'constant')

        audio_path = self._write_temp_audio(audio_array, sampling_rate)

        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": input["instruction"]+"\nPut the result in the following format: \\boxed\{.\}"},
                {"type": "audio", "path": audio_path},
            ]},
        ]

        inputs = self.processor.apply_chat_template(conversation).to(self.model.device, dtype=torch.bfloat16)

        # Generate output
        output = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
        text = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        if input['task_type'] == 'ASR': text = _post_process_voxtral_asr(text)

        return text

    # --- vLLM backend ---

    def load_vllm(self):
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

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        return [
            {"role": "user", "content": [
                _input_audio_part(audio_array, sampling_rate),
                {"type": "text", "text": instruction + "\nPut the result in the following format: \\boxed{.}"},
            ]},
        ]

    def _postprocess_asr_text(self, text):
        return _post_process_voxtral_asr(text)
