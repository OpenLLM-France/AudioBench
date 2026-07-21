import re
import logging

import librosa

from audio_bench.model_src.vllm_model import VLLMModel

logger = logging.getLogger(__name__)


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

class Qwen2Audio7BInstruct(VLLMModel):

    name = "Qwen/Qwen2-Audio-7B-Instruct"

    def __init__(self, gpu_memory_utilization=0.4, device=None):
        super().__init__(model_path="Qwen/Qwen2-Audio-7B-Instruct", gpu_memory_utilization=gpu_memory_utilization, device=device)


    def _infer_single(self, audio_array, sampling_rate, instruction, is_asr):
        """Run inference on a single audio segment."""
        audio_path = self._write_temp_audio(audio_array, sampling_rate)

        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": instruction},
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
        inputs = inputs.to(self.model.device)

        generate_ids = self.model.generate(**inputs, max_length=512)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if is_asr:
            response = _post_process_qwen2_asr(response)

        return response


    # --- VLLM hooks ---

    def _build_vllm_messages(self, audio_array, sampling_rate, instruction):
        from audio_bench.model_src.vllm_backend import _input_audio_part
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                _input_audio_part(audio_array, sampling_rate),
                {"type": "text", "text": instruction},
            ]},
        ]

    def _postprocess_asr_text(self, text):
        return _post_process_qwen2_asr(text)
