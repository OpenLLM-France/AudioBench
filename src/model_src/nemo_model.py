import logging

from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


class NeMoModel(BaseModel):
    """Shared base for models using NeMo's SALM API."""

    def _build_nemo_conversation(self, audio_path, prompt):
        prompt_content = (
            f"{prompt}:\n"
            f"{self.model.audio_locator_tag}\n"
        )
        return [
            {
                "role": "user",
                "content": prompt_content,
                "audio": [audio_path],
            }
        ]

    def _generate(self, input):
        prompt = input["instruction"]

        segments, sampling_rate, mode = self._prepare_audio_segments(
            input["audio"], input['task_type']
        )

        if mode == 'chunked':
            parts = []
            for seg in segments:
                audio_path = self._write_temp_audio(seg, sampling_rate)
                conversation = self._build_nemo_conversation(audio_path, prompt)
                answer_ids = self.model.generate(prompts=[conversation], max_new_tokens=512)
                parts.append(self.model.tokenizer.ids_to_text(answer_ids[0].cpu()))
            return ' '.join(parts)

        audio_path = self._write_temp_audio(segments[0], sampling_rate)
        conversation = self._build_nemo_conversation(audio_path, prompt)
        answer_ids = self.model.generate(prompts=[conversation], max_new_tokens=512)
        return self.model.tokenizer.ids_to_text(answer_ids[0].cpu())

    def _generate_batch(self, inputs):
        all_prompts = []

        for inp in inputs:
            prompt = inp["instruction"]

            segments, sampling_rate, mode = self._prepare_audio_segments(
                inp["audio"], inp['task_type']
            )
            if mode == 'chunked':
                raise RuntimeError(
                    "Audio chunking is not supported with batch_size > 1. "
                    "Use batch_size=1 for long ASR audio."
                )
            elif mode=='padded':
                logger.info(f"Audio was padded: {inp}")
            audio_path = self._write_temp_audio(segments[0], sampling_rate)
            all_prompts.append(self._build_nemo_conversation(audio_path, prompt))

        answer_ids = self.model.generate(prompts=all_prompts, max_new_tokens=512)

        return [
            self.model.tokenizer.ids_to_text(answer_ids[i].cpu())
            for i in range(len(inputs))
        ]
