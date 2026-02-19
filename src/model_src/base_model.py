import torch
import logging

logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all AudioBench models."""

    supports_vllm = False

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_name = None
        self.model_name = None
        self.backend = None

    # --- Public API (called by main_evaluate) ---

    def generate(self, input):
        if self.backend == "vllm":
            if not isinstance(input, list):
                input = [input]
            return self.generate_vllm(input)
        with torch.no_grad():
            return self._generate(input)

    # --- To be implemented by subclasses ---

    def load(self):
        raise NotImplementedError

    def _generate(self, input):
        raise NotImplementedError

    def load_vllm(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not support VLLM backend"
        )

    def generate_vllm(self, inputs):
        raise NotImplementedError(
            f"{type(self).__name__} does not support VLLM backend"
        )
