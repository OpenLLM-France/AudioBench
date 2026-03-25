import logging
import pathlib

import google.generativeai as genai

from audio_bench.model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


def _do_sample_inference(self, audio_array, instruction, sampling_rate=16000):

    audio_path = self._write_temp_audio(audio_array, sampling_rate)

    response = self.model.generate_content([
        instruction,
        {
            "mime_type": "audio/wav",
            "data": pathlib.Path(audio_path).read_bytes()
        }
    ])
    response = response.text
    return response


class Gemini15Flash(BaseModel):

    is_api_model = True

    def load(self):
        # Initialize a Gemini model appropriate for your use case.
        self.model = genai.GenerativeModel('models/gemini-1.5-flash')
        logger.info("Model loaded")

    def _generate(self, input):

        instruction   = input["instruction"]

        segments, sampling_rate, mode = self._prepare_audio_segments(input["audio"], input['task_type'])

        if mode == 'chunked':
            return ' '.join(_do_sample_inference(self, seg, instruction) for seg in segments)
        return _do_sample_inference(self, segments[0], instruction)
