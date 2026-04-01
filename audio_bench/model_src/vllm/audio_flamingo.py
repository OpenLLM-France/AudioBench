import logging
import os
import tempfile
from pathlib import Path

import torch
# from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from audio_bench.model_src.base_model import BaseModel

logger = logging.getLogger(__name__)

def _post_process_flamingo_asr(model_output):
    # Try \boxed{"..."}
    m = re.search(r'\\boxed\{"(.*?)"\}', model_output, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # Try \boxed{...}
    m = re.search(r"\\boxed\{(.+?)\}", model_output)
    if m and m.group(1).strip():
        return m.group(1).strip()

    return model_output

class AudioFlamingo(BaseModel):

    name = "nvidia/audio-flamingo-3-hf"
    supports_vllm = True    # need a very recent version of vllm

    def __init__(self, gpu_memory_utilization=0.4, device=None):
        super().__init__(model_path="nvidia/audio-flamingo-3-hf", gpu_memory_utilization=gpu_memory_utilization, device=device)

    def load(self):
        raise NotImplementedError()
        # self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(self.model_path, device_map=self.device, torch_dtype=torch.bfloat16).eval()
        # self.processor = AutoProcessor.from_pretrained(self.model_path)

    def _generate(self, input):
        audio_array    = input["audio"]["array"]
        sampling_rate  = input["audio"]["sampling_rate"]
        prompt = input["instruction"]+"\nPut the result in the following format: \\boxed\{.\}"

        audio_path = self._write_temp_audio(audio_array, sampling_rate)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=500)

        decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return decoded_outputs

    # --- VLLM backend ---

    def load_vllm(self):
        from vllm import LLM, SamplingParams

        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        self.llm = LLM(
            model=self.model_path,
            max_model_len=4096,
            max_num_seqs=self.batch_size,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=self.gpu_memory_utilization,
            allowed_local_media_path=tempfile.gettempdir(),
        )
        self.sampling_params = SamplingParams(
            temperature=0, max_tokens=4096, repetition_penalty=1.2
        )

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        audio_path = self._write_temp_audio(audio_array, sampling_rate)
        audio_url = Path(audio_path).resolve().as_uri()
        return [
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "audio_url", "audio_url": {"url": audio_url}},
            ]},
        ]
