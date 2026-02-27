import re
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


def _post_process_qwen_asr(model_output):

    match = re.search(r'"((?:\\.|[^"\\])*)"', model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    if ':"' in model_output:
        model_output = '"' + model_output.split(':"')[1]
    elif ': "' in model_output:
        model_output = '"' + model_output.split(': "')[1]

    # Find the longest match of ''
    match = re.search(r'"(.*)"', model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    return model_output


class QwenAudioChat(BaseModel):

    def __init__(self):
        super().__init__(model_path="Qwen/Qwen-Audio-Chat")

    def load(self):
        self.tokenizer               = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model                   = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_path, trust_remote_code=True)
        logger.info(f"Model loaded: {self.model_path}")

    def _infer_single(self, audio_array, sampling_rate, instruction, is_asr):
        """Run inference on a single audio segment."""
        audio_path = self._write_temp_audio(audio_array, sampling_rate)

        query = self.tokenizer.from_list_format([
            {'audio': audio_path},
            {'text': instruction},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)

        if is_asr:
            response = _post_process_qwen_asr(response)

        return response

    def _generate(self, input):

        instruction   = input["instruction"]
        is_asr        = input['task_type'] == 'ASR'

        segments, sampling_rate, mode = self._prepare_audio_segments(input["audio"], input['task_type'])

        if mode == 'chunked':
            return ' '.join(self._infer_single(seg, sampling_rate, instruction, is_asr) for seg in segments)
        return self._infer_single(segments[0], sampling_rate, instruction, is_asr)
