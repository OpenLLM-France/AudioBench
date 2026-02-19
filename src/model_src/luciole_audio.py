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
import nemo.collections.speechlm2 as slm

from model_src.base_model import BaseModel

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


MODEL_PATH = f"{os.getenv('MODELS')}/Canary-Qwen3-1.7B-v2"


class LucioleAudio(BaseModel):

    def load(self):
        self.model = slm.models.SALM.from_pretrained(MODEL_PATH).to(self.device).eval()

    def _generate(self, input):
        audio_array    = input["audio"]["array"]
        sampling_rate  = input["audio"]["sampling_rate"]
        audio_duration = len(audio_array) / sampling_rate
        prompt = input["instruction"]

        os.makedirs('tmp', exist_ok=True)

        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array, sampling_rate)


        prompt_content = (
            f"{prompt}:\n"
            f"{self.model.audio_locator_tag}\n"

        )

        prompts = [
            [
                {
                    "role": "user",
                    "content": prompt_content,
                    "audio": [audio_path.name],
                }
            ]
        ]

        answer_ids = self.model.generate(prompts=prompts, max_new_tokens=512)
        return self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
