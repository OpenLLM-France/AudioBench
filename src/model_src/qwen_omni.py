import re
import logging

import numpy as np
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


def _post_process_qwen2_omni_asr(model_output):
    match = re.search(r"\nassistant\n(.*)", model_output, re.DOTALL)
    if match:
        model_output = match.group(1)

    match = re.search(r"\\boxed\{\"?(.*?)\"?\}", model_output, re.DOTALL)
    if match:
        model_output = match.group(1)
    else:
        model_output = ""

    return model_output

class Qwen2Omni(BaseModel):

    supports_vllm = True

    def __init__(self, model_path="Qwen/Qwen2.5-Omni-3B"):
        super().__init__(model_path=model_path)
        self._asr_text_processor = _post_process_qwen2_omni_asr

    def load(self):
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(self.model_path, device_map="auto")
        self.model.disable_talker()
        logger.info(f"Model loaded: {self.model_path}")

    def _generate(self, input):

        audio_array    = input["audio"]["array"]
        sampling_rate  = input["audio"]["sampling_rate"]
        audio_duration = len(audio_array) / sampling_rate

        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        audio_path = self._write_temp_audio(audio_array, sampling_rate)

        # see https://deepwiki.com/QwenLM/Qwen2.5-Omni/4.1-working-with-audio#4-audio-tasks
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {"role": "user", "content": [
                {"type": "text", "text": input["instruction"]+"\nPut the result in the following format: \\boxed\{.\}"},
                {"type": "audio", "audio": audio_path},
            ]},
        ]

        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt").to(self.model.device).to(self.model.dtype)

        # Generate output
        output = self.model.generate(**inputs, use_audio_in_video=True, return_audio=False)
        text = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if input['task_type'] == 'ASR': text = self._asr_text_processor(text)

        return text

    # --- VLLM hooks ---

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        from model_src.vllm_backend import _input_audio_part
        return [
            {
                "role": "system",
                "content": (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                ),
            },
            {"role": "user", "content": [
                {"type": "text", "text": instruction + "\nPut the result in the following format: \\boxed{.}"},
                _input_audio_part(audio_array, sampling_rate),
            ]},
        ]

    def _postprocess_asr_text(self, text):
        return self._asr_text_processor(text)
