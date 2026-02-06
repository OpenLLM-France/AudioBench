import os

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging

import time

import subprocess
import tempfile
import soundfile as sf
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

# Install fairseq 'pip install --editable ./'


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


model_path = "nvidia/audio-flamingo-3-hf"

def audio_flamingo_model_loader(self):
    self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_path, device_map="auto").eval()
    self.processor = AutoProcessor.from_pretrained(model_path)


def audio_flamingo_model_generation(self, input):
    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate
    prompt = input["instruction"]
    
    os.makedirs('tmp', exist_ok=True)

    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
    sf.write(audio_path.name, audio_array, sampling_rate)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_path.name},
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