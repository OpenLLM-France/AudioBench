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


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

def qwen2_omni_model_loader(self, model_name = "Qwen/Qwen2.5-Omni-3B"):

    self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    self.model.disable_talker()
    logger.info("Model loaded: {}".format(model_name))


def post_process_qwen2_asr(model_output):
    
    match = re.search(r"\nassistant\n(.*)", model_output, re.DOTALL)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    match = re.search(r"\\boxed\{\"?(.*?)\"?\}", model_output, re.DOTALL)
    if match:
        model_output = match.group(1)
    else:
        model_output = ""

    return model_output


def qwen2_omni_model_generation(self, input):

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

    if input['task_type'] == 'ASR': text = post_process_qwen2_asr(text)

    return text

