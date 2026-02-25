import logging

import nemo.collections.speechlm2 as slm

from model_src.nemo_model import NeMoModel

logger = logging.getLogger(__name__)


class CanaryQwen(NeMoModel):

    def __init__(self):
        super().__init__(model_path="nvidia/canary-qwen-2.5b")

    def load(self):
        self.model = slm.models.SALM.from_pretrained(self.model_path).eval()
