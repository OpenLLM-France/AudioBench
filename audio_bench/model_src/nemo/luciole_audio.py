import os
import json
import logging
from pathlib import Path

import nemo.collections.speechlm2 as slm

from audio_bench.model_src.nemo_model import NeMoModel

logger = logging.getLogger(__name__)


def resolve_salm_class(model_path):
    """Pick the SALM class matching the training backend.

    Mirrors resolve_class_path() in the export repo (adapter_training/export/
    export_experiment.py): a checkpoint trained with model.use_nemo_automodel is a
    SALMAutomodel and only that class understands its automodel-style LoRA block
    (alpha/dim -> peft's LoraConfig wants lora_alpha/r, so plain SALM raises
    TypeError: unexpected keyword argument 'alpha'). Everything else stays SALM.

    The flag is read from the exported checkpoint's own config.json, so the eval
    backend can never drift from the one that trained/exported it. A non-local
    model_path (e.g. a HF repo id) has no config.json to read -> SALM, as before.

    NOTE: SALMAutomodel checkpoints ALSO need transformers >= 5.6.0 at runtime (the
    Luciole-8B Nemotron-H backbone is built from mlp blocks, and native nemotron_h only
    gained the "mlp" mixer in 5.6.0). The default audiollm_benchmark:baked image ships
    4.55.2 -> run these on audiollm_benchmark:automodel (evaluate_on_dgx's `image` param).
    """
    config = Path(model_path) / "config.json"
    if not config.is_file():
        return slm.models.SALM
    with open(config) as f:
        use_automodel = bool(json.load(f).get("use_nemo_automodel", False))
    return slm.models.SALMAutomodel if use_automodel else slm.models.SALM


class LucioleAudio(NeMoModel):

    def __init__(self, model_path=None, device=None):
        super().__init__(model_path=model_path, device=device)

    def load(self):
        cls = resolve_salm_class(self.model_path)
        logger.info("Loading %s with %s", self.model_path, cls.__name__)
        self.model = cls.from_pretrained(self.model_path).to(self.torch_device).eval()
