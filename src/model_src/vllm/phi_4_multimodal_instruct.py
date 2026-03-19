import logging
from pathlib import Path

import librosa
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


def _phi4_resample(audio_array, sampling_rate):
    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    return audio_array, sampling_rate


def _do_sample_inference(self, audio_array, prompt):

    audio = [audio_array, 16000]

    inputs = self.processor(text=prompt, audios=[audio], return_tensors='pt').to(self.model.device)
    generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
        )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return response


class Phi4MultimodalInstruct(BaseModel):

    supports_vllm = True

    def __init__(self, gpu_memory_utilization=0.4, device=None):
        super().__init__(model_path="microsoft/Phi-4-multimodal-instruct", gpu_memory_utilization=gpu_memory_utilization, device=device)

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype='auto',
                _attn_implementation='flash_attention_2',
                device_map=self.device,
            ).eval()
        print("model.config._attn_implementation:", self.model.config._attn_implementation)
        self.generation_config = GenerationConfig.from_pretrained(self.model_path, 'generation_config.json')
        logger.info(f"Model loaded: {self.model_path}")

    def _generate(self, input):

        instruction   = input['instruction']

        user_prompt      = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix    = '<|end|>'
        prompt = f'{user_prompt}<|audio_1|>{instruction}{prompt_suffix}{assistant_prompt}'

        segments, sampling_rate, mode = self._prepare_audio_segments(input["audio"], input['task_type'])

        if mode == 'chunked':
            return ' '.join(_do_sample_inference(self, seg, prompt) for seg in segments)
        return _do_sample_inference(self, segments[0], prompt)

    # --- VLLM support ---

    def load_vllm(self):
        from vllm import LLM, SamplingParams
        from huggingface_hub import snapshot_download
        from vllm.lora.request import LoRARequest

        model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
        speech_lora_path = str(Path(model_path) / "speech-lora")

        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=4096,
            max_num_seqs=self.batch_size,
            enable_lora=True,
            max_lora_rank=320,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.lora_request = LoRARequest("speech", 1, speech_lora_path)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=1000)

    # --- VLLM hooks ---

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        from model_src.vllm_backend import _input_audio_part
        return [
            {"role": "user", "content": [
                _input_audio_part(audio_array, sampling_rate),
                {"type": "text", "text": instruction},
            ]},
        ]

    def _preprocess_audio_for_vllm(self, audio_array, sampling_rate):
        return _phi4_resample(audio_array, sampling_rate)

    def _vllm_chat_kwargs(self):
        return {"lora_request": self.lora_request}
