import os
import re
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import librosa
import soundfile as sf
import tempfile

from vllm import LLM, SamplingParams

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =====================================================================
#  Qwen2-Audio-7B-Instruct
# =====================================================================

def qwen2_audio_7b_instruct_vllm_loader(self):
    self.llm = LLM(
        model="Qwen/Qwen2-Audio-7B-Instruct",
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": 1},
    )
    self.sampling_params = SamplingParams(temperature=0, max_tokens=512)


def _post_process_qwen2_asr(model_output):
    match = re.search(r'"((?:\\.|[^"\\])*)"', model_output)
    if match:
        model_output = match.group(1)

    if ":'" in model_output:
        model_output = "'" + model_output.split(":'")[1]
    elif ": '" in model_output:
        model_output = "'" + model_output.split(": '")[1]

    match = re.search(r"'(.*)'", model_output)
    if match:
        model_output = match.group(1)

    return model_output


def _qwen2_audio_vllm_infer_single(self, audio_array, sampling_rate, instruction):
    from vllm import TextPrompt

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>\n"
        f"{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    outputs = self.llm.generate(
        TextPrompt(prompt=prompt, multi_modal_data={"audio": [(audio_array, sampling_rate)]}),
        sampling_params=self.sampling_params,
    )
    return outputs[0].outputs[0].text


def qwen2_audio_7b_instruct_vllm_generation(self, input):
    audio_array = input["audio"]["array"]
    sampling_rate = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate
    instruction = input["instruction"]

    # For ASR task, if audio duration is more than 30 seconds, chunk and infer separately
    if audio_duration > 30 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 30 seconds. Chunking and inferring separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])

        model_predictions = []
        for chunk in audio_chunks:
            response = _qwen2_audio_vllm_infer_single(self, chunk, sampling_rate, instruction)
            response = _post_process_qwen2_asr(response)
            model_predictions.append(response)

        output = ' '.join(model_predictions)

    elif audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')
        audio_array = audio_array[:30 * sampling_rate]
        output = _qwen2_audio_vllm_infer_single(self, audio_array, sampling_rate, instruction)
        if input['task_type'] == 'ASR':
            output = _post_process_qwen2_asr(output)

    else:
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        output = _qwen2_audio_vllm_infer_single(self, audio_array, sampling_rate, instruction)
        if input['task_type'] == 'ASR':
            output = _post_process_qwen2_asr(output)

    return output


# =====================================================================
#  Qwen2.5-Omni (3B / 7B)
# =====================================================================

def qwen2_omni_vllm_loader(self, model_name="Qwen/Qwen2.5-Omni-3B"):
    self.llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": 1},
    )
    self.sampling_params = SamplingParams(temperature=0, max_tokens=512)


def _post_process_qwen2_omni_asr(model_output):
    match = re.search(r"\nassistant\n(.*)", model_output, re.DOTALL)
    if match:
        model_output = match.group(1)

    match = re.search(r"\\boxed\{\"?(.*?)\"?\}", model_output, re.DOTALL)
    if match:
        model_output = match.group(1)
    else:
        model_output = ""

    return model_output


def qwen2_omni_vllm_generation(self, input):
    audio_array = input["audio"]["array"]
    sampling_rate = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate
    instruction = input["instruction"]

    if audio_duration < 1:
        logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
        audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

    from vllm import TextPrompt

    prompt = (
        "<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen Team, "
        "Alibaba Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}\nPut the result in the following format: \\boxed{{.}}\n"
        "<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    outputs = self.llm.generate(
        TextPrompt(prompt=prompt, multi_modal_data={"audio": [(audio_array, sampling_rate)]}),
        sampling_params=self.sampling_params,
    )
    text = outputs[0].outputs[0].text

    if input['task_type'] == 'ASR':
        text = _post_process_qwen2_omni_asr(text)

    return text


# =====================================================================
#  Phi-4-multimodal-instruct
# =====================================================================

def phi_4_multimodal_instruct_vllm_loader(self):
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


def _phi4_vllm_infer_single(self, audio_array, sampling_rate, instruction):
    from vllm import TextPrompt

    prompt = f"<|user|><|audio_1|>{instruction}<|end|><|assistant|>"

    # Phi-4 expects 16kHz audio
    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000

    outputs = self.llm.generate(
        TextPrompt(
            prompt=prompt,
            multi_modal_data={"audio": [(audio_array, sampling_rate)]},
        ),
        sampling_params=self.sampling_params,
        lora_request=self.lora_request,
    )
    return outputs[0].outputs[0].text


def phi_4_multimodal_instruct_vllm_generation(self, input):
    audio_array = input["audio"]["array"]
    sampling_rate = input["audio"]["sampling_rate"]
    instruction = input["instruction"]
    audio_duration = len(audio_array) / sampling_rate

    # For ASR task, if audio duration is more than 40 seconds, chunk and infer separately
    if audio_duration > 40 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 40 seconds. Chunking and inferring separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 40 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 40 * sampling_rate])

        model_predictions = [_phi4_vllm_infer_single(self, chunk, sampling_rate, instruction) for chunk in audio_chunks]
        output = ' '.join(model_predictions)

    elif audio_duration > 40:
        logger.info('Audio duration is more than 40 seconds. Taking first 40 seconds.')
        audio_array = audio_array[:40 * sampling_rate]
        output = _phi4_vllm_infer_single(self, audio_array, sampling_rate, instruction)

    else:
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        output = _phi4_vllm_infer_single(self, audio_array, sampling_rate, instruction)

    return output


# =====================================================================
#  Whisper Large v3
# =====================================================================

def whisper_large_v3_vllm_loader(self):
    self.llm = LLM(
        model="openai/whisper-large-v3",
        max_model_len=448,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": 1},
    )
    self.sampling_params = SamplingParams(temperature=0, max_tokens=448)


def whisper_large_v3_vllm_generation(self, sample):
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]

    from vllm import TextPrompt

    # Language/task tokens for Whisper
    if sample['task_type'] == 'ASR':
        prompt = "<|startoftranscript|>"
    elif sample['task_type'] == 'ASR-ZH':
        prompt = "<|startoftranscript|><|zh|><|transcribe|>"
    elif sample['task_type'] in ["ST-ID-EN", "ST-TA-EN", "ST-ZH-EN"]:
        prompt = "<|startoftranscript|><|en|><|translate|>"
    else:
        raise NotImplementedError(f"Whisper does not support other task: {sample['task_type']}.")

    outputs = self.llm.generate(
        TextPrompt(prompt=prompt, multi_modal_data={"audio": [(audio_array, sampling_rate)]}),
        sampling_params=self.sampling_params,
    )
    return outputs[0].outputs[0].text.strip()


# =====================================================================
#  Whisper Large v2
# =====================================================================

def whisper_large_v2_vllm_loader(self):
    self.llm = LLM(
        model="openai/whisper-large-v2",
        max_model_len=448,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": 1},
    )
    self.sampling_params = SamplingParams(temperature=0, max_tokens=448)


def whisper_large_v2_vllm_generation(self, sample):
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]

    from vllm import TextPrompt

    if sample['task_type'] == 'ASR':
        prompt = "<|startoftranscript|>"
    elif sample['task_type'] == 'ASR-ZH':
        prompt = "<|startoftranscript|><|zh|><|transcribe|>"
    elif sample['task_type'] in ["ST-ID-EN", "ST-TA-EN", "ST-ZH-EN"]:
        prompt = "<|startoftranscript|><|en|><|translate|>"
    else:
        raise NotImplementedError(f"Whisper does not support other task: {sample['task_type']}.")

    outputs = self.llm.generate(
        TextPrompt(prompt=prompt, multi_modal_data={"audio": [(audio_array, sampling_rate)]}),
        sampling_params=self.sampling_params,
    )
    return outputs[0].outputs[0].text.strip()
