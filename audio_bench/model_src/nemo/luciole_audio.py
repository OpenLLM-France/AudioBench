import os
import logging
from pathlib import Path

import nemo.collections.speechlm2 as slm

from audio_bench.model_src.nemo_model import NeMoModel

logger = logging.getLogger(__name__)


class LucioleAudio(NeMoModel):

    def __init__(self, model_path=None, device=None):
        super().__init__(model_path=model_path, device=device)

    def load(self):
        self.model = slm.models.SALM.from_pretrained(self.model_path).to(self.torch_device).eval()
