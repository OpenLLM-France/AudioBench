import logging

from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


class AudioFlamingo(BaseModel):

    def __init__(self):
        super().__init__(model_path="nvidia/audio-flamingo-3-hf")

    def load(self):
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(self.model_path, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def _generate(self, input):
        audio_array    = input["audio"]["array"]
        sampling_rate  = input["audio"]["sampling_rate"]
        prompt = input["instruction"]

        audio_path = self._write_temp_audio(audio_array, sampling_rate)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": audio_path},
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
