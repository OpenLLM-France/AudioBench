import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


def _do_sample_inference(self, audio_array, instruction):

    prompt = f"Given the following audio context: <SpeechHere>\n\nText instruction: {instruction}"
    conversation = [
            {"role": "user", "content": prompt}
        ]

    chat_prompt = self.processor.tokenizer.apply_chat_template(
                conversation          = conversation,
                tokenize              = False,
                add_generation_prompt = True
            )

    inputs = self.processor(text=chat_prompt, audios=audio_array)

    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to('cuda')
        if inputs[key].dtype is torch.float32:
            inputs[key] = inputs[key].to(torch.bfloat16)

    model_outputs = self.model.generate(**inputs, max_new_tokens=228)
    generated_ids = model_outputs[:, inputs['input_ids'].size(1):]
    response      = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


class MeralionAudioLLMWhisperSeaLion(BaseModel):

    def __init__(self):
        super().__init__(model_path="MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION")

    def load(self):
        self.processor = AutoProcessor.from_pretrained(
        self.model_path,
        trust_remote_code=True,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            use_safetensors=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        )
        self.model.to("cuda")

        logger.info(f"Model loaded: {self.model_path}")

    def _generate(self, input):

        audio_array   = input["audio"]["array"]
        sampling_rate = input["audio"]["sampling_rate"]
        instruction   = input["instruction"]

        segments, mode = self._prepare_audio_segments(audio_array, sampling_rate, input['task_type'])

        if mode == 'chunked':
            return ' '.join(_do_sample_inference(self, seg, instruction) for seg in segments)
        return _do_sample_inference(self, segments[0], instruction)
