import logging

import torch
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


def _response_to_audio(conversation, model=None, processor=None):
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if ele['audio_url'] != None:
                        audios.append(librosa.load(
                            ele['audio_url'],
                            sr=processor.feature_extractor.sampling_rate)[0]
                        )
    if audios != []:
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True,sampling_rate=16000)
    else:
        inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to(model.device)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
    generate_ids = model.generate(**inputs, max_new_tokens=2048, temperature = 0, do_sample=False)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


def _do_sample_inference(self, audio_array, prompt):

    audio_path = self._write_temp_audio(audio_array, 16000)

    # Audio Analysis
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt},
        ]},
    ]

    response = _response_to_audio(conversation, model=self.model, processor=self.processor)

    return response


class SeallmsAudio7B(BaseModel):

    max_audio_duration = 40
    supports_vllm = True

    def __init__(self, gpu_memory_utilization=0.4):
        super().__init__(model_path="SeaLLMs/SeaLLMs-Audio-7B", gpu_memory_utilization=gpu_memory_utilization)

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.bfloat16).eval()
        logger.info(f"Model loaded: {self.model_path}")

    def _vllm_chat_kwargs(self):
        return {"chat_template_content_format": "string"}

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        from model_src.vllm_backend import _input_audio_part
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                _input_audio_part(audio_array, sampling_rate),
                {"type": "text", "text": instruction},
            ]},
        ]

    def _generate(self, input):

        instruction   = input['instruction']

        segments, sampling_rate, mode = self._prepare_audio_segments(input["audio"], input['task_type'])

        if mode == 'chunked':
            return ' '.join(_do_sample_inference(self, seg, instruction) for seg in segments)
        return _do_sample_inference(self, segments[0], instruction)