import os
import logging
from pathlib import Path

import nemo.collections.speechlm2 as slm

from model_src.nemo_model import NeMoModel

logger = logging.getLogger(__name__)


class LucioleAudio(NeMoModel):

    def __init__(self, model_path=None):
        super().__init__(model_path=model_path)

    def load(self):
        self.model = slm.models.SALM.from_pretrained(self.model_path).to(self.device).eval()
