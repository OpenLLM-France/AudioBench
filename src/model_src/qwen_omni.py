import os
import re

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch

from tqdm import tqdm

import soundfile as sf

from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

import tempfile

from model_src.base_model import BaseModel


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


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
        super().__init__()
        self._model_path = model_path

    def load(self):
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self._model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(self._model_path, device_map="auto")
        self.model.disable_talker()
        logger.info("Model loaded: {}".format(self._model_path))

    def _generate(self, input):

        audio_array    = input["audio"]["array"]
        sampling_rate  = input["audio"]["sampling_rate"]
        audio_duration = len(audio_array) / sampling_rate

        os.makedirs('tmp', exist_ok=True)

        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array, sampling_rate)

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
                {"type": "audio", "audio": audio_path.name},
            ]},
        ]

        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt").to(self.model.device).to(self.model.dtype)

        # Generate output
        output = self.model.generate(**inputs, use_audio_in_video=True, return_audio=False)
        text = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if input['task_type'] == 'ASR': text = _post_process_qwen2_omni_asr(text)

        return text

    # --- VLLM support ---

    def load_vllm(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=self._model_path,
            max_model_len=4096,
            max_num_seqs=5,
            limit_mm_per_prompt={"audio": 1},
        )
        self.sampling_params = SamplingParams(temperature=0, max_tokens=512)

    def generate_vllm(self, inputs):
        from model_src.vllm_backend import _input_audio_part

        all_messages = []
        for inp in inputs:
            audio_array = inp["audio"]["array"]
            sampling_rate = inp["audio"]["sampling_rate"]
            audio_duration = len(audio_array) / sampling_rate

            if audio_duration < 1:
                audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

            all_messages.append(_build_vllm_messages(audio_array, sampling_rate, inp["instruction"]))

        outputs = self.llm.chat(all_messages, sampling_params=self.sampling_params)

        results = []
        for output, inp in zip(outputs, inputs):
            text = output.outputs[0].text
            if inp['task_type'] == 'ASR':
                text = _post_process_qwen2_omni_asr(text)
            results.append(text)

        return results


def _build_vllm_messages(audio_array, sampling_rate, instruction):
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
