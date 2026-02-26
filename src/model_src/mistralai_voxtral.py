import re
import logging

import numpy as np
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor

from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


def _post_process_voxtral_asr(model_output):

    match = re.search(r"\n\n(.*)", model_output, re.DOTALL)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    match = re.search(r"\\boxed\{\\text\{?(.*?)\}?\}", model_output, re.DOTALL)
    if match:
        model_output = match.group(1)
    else:
        model_output = ""

    return model_output


class Voxtral(BaseModel):

    def __init__(self, model_path="mistralai/Voxtral-Mini-3B-2507"):
        super().__init__(model_path=model_path)

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = VoxtralForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
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
