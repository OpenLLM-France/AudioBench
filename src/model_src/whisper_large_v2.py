import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


class WhisperLargeV2(BaseModel):

    supports_vllm = True

    def __init__(self):
        super().__init__(model_path="openai/whisper-large-v2")

    def load(self):
        self.whisper_model     = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True, device_map="auto")
        self.whisper_processor = AutoProcessor.from_pretrained(self.model_path)
        self.whisper_pipe      = pipeline(
                        "automatic-speech-recognition",
                        model              = self.whisper_model,
                        tokenizer          = self.whisper_processor.tokenizer,
                        feature_extractor  = self.whisper_processor.feature_extractor,
                        max_new_tokens     = 128,
                        chunk_length_s     = 30,
                        batch_size         = 16,
                        return_timestamps  = False,
                        torch_dtype        = torch.float16,
                        device_map         = "auto",
                    )
        self.whisper_model.eval()

        logger.info(f"Model loaded from {self.model_path}.")

    def _generate(self, sample):

        if sample['task_type'] == 'ASR':
            whisper_output = self.whisper_pipe(sample['audio'])['text'].strip()
            return whisper_output

        elif sample['task_type'] == "ASR-ZH":
            whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"language": "zh"})['text'].strip()
            return whisper_output

        elif sample['task_type'] in ["ST-ID-EN",
                                     "ST-TA-EN",
                                     "ST-ZH-EN",
                                     ]:
            whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"task": "translate", "language": "en"})['text'].strip()
            return whisper_output

        else:
            raise NotImplementedError(f"Whisper does not support other task: {sample['task_type']}.")

    # --- VLLM support ---

    def load_vllm(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=self.model_path,
            max_model_len=448,
            max_num_seqs=5,
            limit_mm_per_prompt={"audio": 1},
        )
        self.sampling_params = SamplingParams(temperature=0, max_tokens=448)

    def generate_vllm(self, samples):
        from vllm import TextPrompt
        from model_src.vllm_backend import _whisper_task_prompt

        all_prompts = [
            TextPrompt(
                prompt=_whisper_task_prompt(s['task_type']),
                multi_modal_data={"audio": [(s["audio"]["array"], s["audio"]["sampling_rate"])]},
            )
            for s in samples
        ]
        outputs = self.llm.generate(all_prompts, sampling_params=self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]
