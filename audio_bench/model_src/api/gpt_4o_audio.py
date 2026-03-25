import io
import os
import base64
import logging
import time

import soundfile as sf
from openai import AzureOpenAI

from audio_bench.model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


def _do_sample_inference(self, audio_array, instruction, sampling_rate=16000):

    # Create an in-memory buffer
    buffer = io.BytesIO()

    # Write the WAV data to the buffer
    sf.write(buffer, audio_array, sampling_rate, format='WAV')

    # Get the byte data from buffer
    wav_data = buffer.getvalue()

    # Encode to Base64
    encoded_string = base64.b64encode(wav_data).decode('utf-8')

    chat_prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": instruction
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_string,
                        "format": "wav"
                    }
                }
            ]
        }
    ]

    # Include speech result if speech is enabled
    messages = chat_prompt

    try:

        completion = self.client.chat.completions.create(
                model             = self.deployment,
                messages          = messages,
                max_tokens        = 5000,
                temperature       = 0.7,
                top_p             = 0.95,
                frequency_penalty = 0,
                presence_penalty  = 0,
                stop              = None,
                stream            = False
            )

        response = completion.choices[0].message.content

    except:
        print("Some error happened to GPT-4o-Audio model, stop the inference.")
        response = "Dummy model generation."
        time.sleep(2)

    return response


class GPT4oAudio(BaseModel):

    is_api_model = True

    def load(self):
        endpoint         = os.getenv("ENDPOINT_URL", "https://aoai-i2r-test-001.openai.azure.com/")
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")
        self.deployment  = os.getenv("DEPLOYMENT_NAME", "gpt-4o-audio-preview")

        # Initialize Azure OpenAI Service client with key-based authentication
        self.client = AzureOpenAI(
                azure_endpoint = endpoint,
                api_key        = subscription_key,
                api_version    = "2024-11-01-preview",
            )

        logger.info("Model loaded")

    def _generate(self, input):

        instruction   = input["instruction"]

        segments, sampling_rate, mode = self._prepare_audio_segments(input["audio"], input['task_type'])

        if mode == 'chunked':
            return ' '.join(_do_sample_inference(self, seg, instruction) for seg in segments)
        return _do_sample_inference(self, segments[0], instruction)
