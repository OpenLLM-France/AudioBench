import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM

from audio_bench.model_src.base_model import BaseModel

logger = logging.getLogger(__name__)

class WhisperLargeV3WithLlama38BInstruct(BaseModel):

    def __init__(self, device=None):
        super().__init__(model_path="openai/whisper-large-v3", device=device)
        self.llm_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

    def load(self):
        self.whisper_model     = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True, device_map=self.device)
        self.whisper_processor = AutoProcessor.from_pretrained(self.model_path)
        self.whisper_pipe      = pipeline(
                        "automatic-speech-recognition",
                        model=self.whisper_model,
                        tokenizer=self.whisper_processor.tokenizer,
                        feature_extractor=self.whisper_processor.feature_extractor,
                        max_new_tokens=128,
                        chunk_length_s=30,
                        batch_size=16,
                        return_timestamps=True,
                        torch_dtype=torch.float16,
                        device_map=self.device,
                    )
        self.whisper_model.eval()

        self.llm_tokenizer           = AutoTokenizer.from_pretrained(self.llm_model_path, padding_side='left')
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model               = AutoModelForCausalLM.from_pretrained(self.llm_model_path, device_map=self.device, torch_dtype=torch.float16)
        self.llm_model.eval()

        logger.info(f"Model loaded from {self.model_path} and {self.llm_model_path}.")

    def _generate(self, sample):

        if sample['task_type'] == 'ASR' and sample.get('language') == 'ZH':
            whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"language": "zh"})['text'].strip()
            return whisper_output

        elif sample['task_type'] == 'ASR':
            whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"language": "en"})['text'].strip()
            return whisper_output

        elif sample['task_type'] == 'AST':
            whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"task": "translate", "language": "en"})['text'].strip()
            return whisper_output

        else:
            whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"language": "en"})['text'].strip()

            instruction = sample['instruction']

            prompt = f"""\
            [Audio Transcriptions]
            {whisper_output}

            [Question]
            {instruction}

            [System]
            Please answer the instruction based on the audio transcription provided above.
            Ensure that your response adheres to the following format:

            Answer: (Provide a precise and concise answer here.)
            """
            batch_input = [prompt]

            # If instruction following task, then only use whisper_output
            if sample['task_type'] == "Instruction Following":
                batch_input = [whisper_output]

            batch_input_templated = []
            for sample in batch_input:
                messages = [
                    {"role": "user", "content": sample},
                ]
                sample_templated = self.llm_tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)
                batch_input_templated.append(sample_templated)

            batch_input = batch_input_templated

            encoded_batch        = self.llm_tokenizer(batch_input, return_tensors="pt", padding=True).to(self.llm_model.device)
            generated_ids        = self.llm_model.generate(**encoded_batch, max_new_tokens=500, pad_token_id=self.llm_tokenizer.eos_token_id)
            generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
            decoded_batch_output = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if 'Answer: ' in decoded_batch_output:
                decoded_batch_output = decoded_batch_output.split('Answer: ')[1].strip()

            return decoded_batch_output
