# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM

from model_src.base_model import BaseModel

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

WHISPER_MODEL_PATH = "openai/whisper-large-v3"


class WhisperLargeV3(BaseModel):

    supports_vllm = True

    def load(self):
        self.whisper_model     = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True, device_map="auto")
        self.whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_PATH)
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

        logging.info(f"Model loaded from {WHISPER_MODEL_PATH}.")

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
            model="openai/whisper-large-v3",
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
