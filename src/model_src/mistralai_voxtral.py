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
from transformers import VoxtralForConditionalGeneration, AutoProcessor, GenerationConfig

import tempfile


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

def voxtral_model_loader(self, model_name = "mistralai/Voxtral-Mini-3B-2507"):

    self.processor = AutoProcessor.from_pretrained(model_name)
    self.model = VoxtralForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    logger.info("Model loaded: {}".format(model_name))


def post_process_voxtral_asr(model_output):

    match = re.search(r"\n\n(.*)", model_output, re.DOTALL)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    match = re.search(r"\\boxed\{\\text\{?(.*?)\}?\}", model_output, re.DOTALL)
    # match = re.search(r"\\boxed\{(.*?)\}", model_output, re.DOTALL)
    if match:
        model_output = match.group(1)
    else:
        model_output = ""

    return model_output

def voxtral_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate

    os.makedirs('tmp', exist_ok=True)

    if audio_duration < 1:
        logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
        audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
    sf.write(audio_path.name, audio_array, sampling_rate)

    conversation = [
        {"role": "user", "content": [
            {"type": "text", "text": input["instruction"]+"\nPut the result in the following format: \\boxed\{.\}"},
            {"type": "audio", "path": audio_path.name},
        ]},
    ]
    
    inputs = self.processor.apply_chat_template(conversation).to(self.model.device, dtype=torch.bfloat16)

    # Generate output
    output = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
    text = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    if input['task_type'] == 'ASR': text = post_process_voxtral_asr(text)

    return text

