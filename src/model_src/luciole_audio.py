import os
import logging

import nemo.collections.speechlm2 as slm

from model_src.nemo_model import NeMoModel

logger = logging.getLogger(__name__)


class LucioleAudio(NeMoModel):

    def __init__(self):
        super().__init__(model_path=f"{os.getenv('MODELS')}/Canary-Qwen3-1.7B-v2")

    def load(self):
        self.model = slm.models.SALM.from_pretrained(self.model_path).to(self.device).eval()
