import logging

import nemo.collections.speechlm2 as slm

from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


class CanaryQwen(BaseModel):

    def __init__(self):
        super().__init__(model_path="nvidia/canary-qwen-2.5b")

    def load(self):
        self.model = slm.models.SALM.from_pretrained(self.model_path).eval()

    def _generate(self, input):
        audio_array    = input["audio"]["array"]
        sampling_rate  = input["audio"]["sampling_rate"]
        prompt = input["instruction"]

        audio_path = self._write_temp_audio(audio_array, sampling_rate)

        prompt_content = (
            f"{prompt}:\n"
            f"{self.model.audio_locator_tag}\n"
        )

        prompts = [
            [
                {
                    "role": "user",
                    "content": prompt_content,
                    "audio": [audio_path],
                }
            ]
        ]

        answer_ids = self.model.generate(prompts=prompts, max_new_tokens=512)
        return self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
