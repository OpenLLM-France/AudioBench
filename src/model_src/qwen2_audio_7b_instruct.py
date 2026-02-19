import os
import re

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch

from tqdm import tqdm

import soundfile as sf

from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

import tempfile

from model_src.base_model import BaseModel


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"


def _post_process_qwen2_asr(model_output):

    match = re.search(r'"((?:\\.|[^"\\])*)"', model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    if ":'" in model_output:
        model_output = "'" + model_output.split(":'")[1]
    elif ": '" in model_output:
        model_output = "'" + model_output.split(": '")[1]

    # Find the longest match of ''
    match = re.search(r"'(.*)'", model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    return model_output


class Qwen2Audio7BInstruct(BaseModel):

    supports_vllm = True

    def load(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_PATH, device_map="auto")
        logger.info("Model loaded: {}".format(MODEL_PATH))

    def _generate(self, input):

        audio_array    = input["audio"]["array"]
        sampling_rate  = input["audio"]["sampling_rate"]
        audio_duration = len(audio_array) / sampling_rate

        os.makedirs('tmp', exist_ok=True)

        # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
        if audio_duration > 30 and input['task_type'] == 'ASR':
            logger.info('Audio duration is more than 30 seconds. Chunking and inferring separately.')
            audio_chunks = []
            for i in range(0, len(audio_array), 30 * sampling_rate):
                audio_chunks.append(audio_array[i:i + 30 * sampling_rate])

            model_predictions = []
            for chunk in tqdm(audio_chunks):
                audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
                sf.write(audio_path.name, chunk, sampling_rate)

                conversation = [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {"role": "user", "content": [
                        {"type": "audio", "audio_url": audio_path.name},
                        {"type": "text", "text": input["instruction"]},
                    ]},
                ]

                text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                audios = []
                for message in conversation:
                    if isinstance(message["content"], list):
                        for ele in message["content"]:
                            if ele["type"] == "audio":
                                audios.append(
                                    librosa.load(
                                        ele['audio_url'],
                                        sr=self.processor.feature_extractor.sampling_rate)[0]
                                )

                inputs = self.processor(text=text, audios=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
                inputs = inputs.to("cuda")

                generate_ids = self.model.generate(**inputs, max_length=512)
                generate_ids = generate_ids[:, inputs.input_ids.size(1):]

                response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                # Reprocess the results to get the output
                if input['task_type'] == 'ASR': response = _post_process_qwen2_asr(response)

                model_predictions.append(response)

            output = ' '.join(model_predictions)

        elif audio_duration > 30:
            logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')
            audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
            sf.write(audio_path.name, audio_array[:30 * sampling_rate], sampling_rate)

            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_path.name},
                    {"type": "text", "text": input["instruction"]},
                ]},
            ]

            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(
                                    ele['audio_url'],
                                    sr=self.processor.feature_extractor.sampling_rate)[0]
                            )

            inputs = self.processor(text=text, audios=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
            inputs = inputs.to("cuda")

            generate_ids = self.model.generate(**inputs, max_length=512)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]

            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            output = response

            # Reprocess the results to get the output
            if input['task_type'] == 'ASR': output = _post_process_qwen2_asr(output)


        else:
            if audio_duration < 1:
                logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
                audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

            audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
            sf.write(audio_path.name, audio_array, sampling_rate)

            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_path.name},
                    {"type": "text", "text": input["instruction"]},
                ]},
            ]

            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(
                                    ele['audio_url'],
                                    sr=self.processor.feature_extractor.sampling_rate)[0]
                            )

            inputs = self.processor(text=text, audios=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
            inputs = inputs.to("cuda")

            generate_ids = self.model.generate(**inputs, max_length=512)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]

            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            output = response

            # Reprocess the results to get the output
            if input['task_type'] == 'ASR': output = _post_process_qwen2_asr(output)

        return output

    # --- VLLM support ---

    def load_vllm(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model="Qwen/Qwen2-Audio-7B-Instruct",
            max_model_len=4096,
            max_num_seqs=5,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=0.6,
        )
        self.sampling_params = SamplingParams(temperature=0, max_tokens=512)

    def generate_vllm(self, inputs):
        from model_src.vllm_backend import _input_audio_part

        results = [None] * len(inputs)
        all_messages = []
        meta = []
        chunk_buffers = {}

        for i, inp in enumerate(inputs):
            audio_array = inp["audio"]["array"]
            sampling_rate = inp["audio"]["sampling_rate"]
            audio_duration = len(audio_array) / sampling_rate
            instruction = inp["instruction"]

            if audio_duration > 30 and inp['task_type'] == 'ASR':
                chunk_buffers[i] = []
                for j in range(0, len(audio_array), 30 * sampling_rate):
                    chunk = audio_array[j:j + 30 * sampling_rate]
                    all_messages.append(_build_vllm_messages(chunk, sampling_rate, instruction))
                    meta.append(('chunk', i))

            elif audio_duration > 30:
                audio_array = audio_array[:30 * sampling_rate]
                all_messages.append(_build_vllm_messages(audio_array, sampling_rate, instruction))
                meta.append(('normal', i, inp['task_type'] == 'ASR'))

            else:
                if audio_duration < 1:
                    audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')
                all_messages.append(_build_vllm_messages(audio_array, sampling_rate, instruction))
                meta.append(('normal', i, inp['task_type'] == 'ASR'))

        outputs = self.llm.chat(all_messages, sampling_params=self.sampling_params)

        for output, m in zip(outputs, meta):
            text = output.outputs[0].text
            if m[0] == 'chunk':
                chunk_buffers[m[1]].append(_post_process_qwen2_asr(text))
            else:
                _, result_idx, apply_asr = m
                if apply_asr:
                    text = _post_process_qwen2_asr(text)
                results[result_idx] = text

        for result_idx, chunks in chunk_buffers.items():
            results[result_idx] = ' '.join(chunks)

        return results


def _build_vllm_messages(audio_array, sampling_rate, instruction):
    from model_src.vllm_backend import _input_audio_part
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            _input_audio_part(audio_array, sampling_rate),
            {"type": "text", "text": instruction},
        ]},
    ]
