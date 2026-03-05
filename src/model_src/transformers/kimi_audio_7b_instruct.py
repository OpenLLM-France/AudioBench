import logging
from model_src.base_model import BaseModel

logger = logging.getLogger(__name__)

SAMPLING_PARAMS = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}


class KimiAudio7BInstruct(BaseModel):

    def __init__(self, model_path=None):
        super().__init__(model_path=model_path or "moonshotai/Kimi-Audio-7B-Instruct")

    def load(self):
        from kimia_infer.api.kimia import KimiAudio
        self.model = KimiAudio(model_path=self.model_path, load_detokenizer=False)
        logger.info(f"Model loaded: {self.model_path}")

    def _generate(self, input):
        instruction = input["instruction"]
        segments, sampling_rate, mode = self._prepare_audio_segments(
            input["audio"], input['task_type']
        )

        if mode == 'chunked':
            return ' '.join(
                self._infer_single(seg, sampling_rate, instruction)
                for seg in segments
            )
        return self._infer_single(segments[0], sampling_rate, instruction)

    def _infer_single(self, audio_array, sampling_rate, instruction):
        import tqdm as tqdm_module
        audio_path = self._write_temp_audio(audio_array, sampling_rate)
        messages = [
            {"role": "user", "message_type": "text", "content": instruction},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]
        original_tqdm = tqdm_module.tqdm
        tqdm_module.tqdm = lambda *args, **kwargs: original_tqdm(*args, **{**kwargs, "disable": True})
        try:
            _, text_output = self.model.generate(
                messages, **SAMPLING_PARAMS, output_type="text"
            )
        finally:
            tqdm_module.tqdm = original_tqdm
        return text_output
