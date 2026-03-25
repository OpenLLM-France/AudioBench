import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from audio_bench.model_src.base_model import BaseModel

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
            inputs[key] = inputs[key].to(self.model.device)
        if inputs[key].dtype is torch.float32:
            inputs[key] = inputs[key].to(torch.bfloat16)

    model_outputs = self.model.generate(**inputs, max_new_tokens=228)
    generated_ids = model_outputs[:, inputs['input_ids'].size(1):]
    response      = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


class MeralionAudioLLMWhisperSeaLion(BaseModel):

    def __init__(self, device=None):
        super().__init__(model_path="MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION", device=device)

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
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()

        logger.info(f"Model loaded: {self.model_path}")

    def _generate(self, input):

        instruction   = input["instruction"]

        segments, sampling_rate, mode = self._prepare_audio_segments(input["audio"], input['task_type'])

        if mode == 'chunked':
            return ' '.join(_do_sample_inference(self, seg, instruction) for seg in segments)
        return _do_sample_inference(self, segments[0], instruction)

    def _generate_batch(self, inputs):
        all_texts = []
        all_audios = []

        for inp in inputs:
            instruction = inp["instruction"]

            segments, sampling_rate, mode = self._prepare_audio_segments(
                inp["audio"], inp['task_type']
            )
            if mode == 'chunked':
                raise RuntimeError(
                    "Audio chunking is not supported with batch_size > 1. "
                    "Use batch_size=1 for long ASR audio."
                )

            prompt = f"Given the following audio context: <SpeechHere>\n\nText instruction: {instruction}"
            conversation = [{"role": "user", "content": prompt}]
            chat_prompt = self.processor.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            all_texts.append(chat_prompt)
            all_audios.append(segments[0])

        batch_inputs = self.processor(text=all_texts, audios=all_audios)
        for key in batch_inputs:
            if isinstance(batch_inputs[key], torch.Tensor):
                batch_inputs[key] = batch_inputs[key].to(self.model.device)
                if batch_inputs[key].dtype is torch.float32:
                    batch_inputs[key] = batch_inputs[key].to(torch.bfloat16)

        model_outputs = self.model.generate(**batch_inputs, max_new_tokens=228)
        generated_ids = model_outputs[:, batch_inputs['input_ids'].size(1):]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
