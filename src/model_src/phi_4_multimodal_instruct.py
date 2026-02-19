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
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

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

MODEL_PATH = "microsoft/Phi-4-multimodal-instruct"


def _phi4_resample(audio_array, sampling_rate):
    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    return audio_array, sampling_rate


def _do_sample_inference(self, audio_array, prompt):

    audio = [audio_array, 16000]

    inputs = self.processor(text=prompt, audios=[audio], return_tensors='pt').to('cuda:0')
    generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
        )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return response


class Phi4MultimodalInstruct(BaseModel):

    supports_vllm = True

    def load(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                torch_dtype='auto',
                _attn_implementation='flash_attention_2',
            ).cuda()
        print("model.config._attn_implementation:", self.model.config._attn_implementation)
        self.generation_config = GenerationConfig.from_pretrained(MODEL_PATH, 'generation_config.json')
        logger.info("Model loaded: {}".format(MODEL_PATH))

    def _generate(self, input):

        audio_array    = input["audio"]["array"]
        sampling_rate  = input["audio"]["sampling_rate"]
        instruction    = input['instruction']
        audio_duration = len(audio_array) / sampling_rate

        user_prompt      = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix    = '<|end|>'
        prompt = f'{user_prompt}<|audio_1|>{instruction}{prompt_suffix}{assistant_prompt}'


        # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
        if audio_duration > 40 and input['task_type'] == 'ASR':
            logger.info('Audio duration is more than 40 seconds. Chunking and inferring separately.')
            audio_chunks = []
            for i in range(0, len(audio_array), 40 * sampling_rate):
                audio_chunks.append(audio_array[i:i + 40 * sampling_rate])

            model_predictions = [_do_sample_inference(self, chunk_array, prompt) for chunk_array in tqdm(audio_chunks)]
            output = ' '.join(model_predictions)


        elif audio_duration > 40:
            logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')

            audio_array = audio_array[:40 * sampling_rate]
            output = _do_sample_inference(self, audio_array, prompt)

        else:
            if audio_duration < 1:
                logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
                audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

            output = _do_sample_inference(self, audio_array, prompt)

        return output

    # --- VLLM support ---

    def load_vllm(self):
        from vllm import LLM, SamplingParams
        from huggingface_hub import snapshot_download
        from vllm.lora.request import LoRARequest

        model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
        speech_lora_path = os.path.join(model_path, "speech-lora")

        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=12800,
            max_num_seqs=2,
            enable_lora=True,
            max_lora_rank=320,
            limit_mm_per_prompt={"audio": 1},
        )
        self.lora_request = LoRARequest("speech", 1, speech_lora_path)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=1000)

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

            if audio_duration > 40 and inp['task_type'] == 'ASR':
                chunk_buffers[i] = []
                for j in range(0, len(audio_array), 40 * sampling_rate):
                    chunk = audio_array[j:j + 40 * sampling_rate]
                    chunk, sr = _phi4_resample(chunk, sampling_rate)
                    all_messages.append(_build_vllm_messages(chunk, sr, instruction))
                    meta.append(('chunk', i))

            elif audio_duration > 40:
                audio_array = audio_array[:40 * sampling_rate]
                audio_array, sampling_rate = _phi4_resample(audio_array, sampling_rate)
                all_messages.append(_build_vllm_messages(audio_array, sampling_rate, instruction))
                meta.append(('normal', i))

            else:
                if audio_duration < 1:
                    audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')
                audio_array, sampling_rate = _phi4_resample(audio_array, sampling_rate)
                all_messages.append(_build_vllm_messages(audio_array, sampling_rate, instruction))
                meta.append(('normal', i))

        outputs = self.llm.chat(
            all_messages,
            sampling_params=self.sampling_params,
            lora_request=self.lora_request,
        )

        for output, m in zip(outputs, meta):
            text = output.outputs[0].text
            if m[0] == 'chunk':
                chunk_buffers[m[1]].append(text)
            else:
                results[m[1]] = text

        for result_idx, chunks in chunk_buffers.items():
            results[result_idx] = ' '.join(chunks)

        return results


def _build_vllm_messages(audio_array, sampling_rate, instruction):
    from model_src.vllm_backend import _input_audio_part
    return [
        {"role": "user", "content": [
            _input_audio_part(audio_array, sampling_rate),
            {"type": "text", "text": instruction},
        ]},
    ]
