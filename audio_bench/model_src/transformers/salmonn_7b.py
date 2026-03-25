import logging

import numpy as np

from SALMONN_7B.model import SALMONN

from audio_bench.model_src.base_model import BaseModel

logger = logging.getLogger(__name__)


class Salmonn7B(BaseModel):

    def __init__(self, device=None):
        super().__init__(model_path="examples/SALMONN_7B/", device=device)

    def load(self):
        self.model = SALMONN(
            ckpt         = self.model_path + "ckpt_path/salmonn_7b_v0.pth",
            whisper_path = self.model_path + "whisper",
            beats_path   = self.model_path + "beats_path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            vicuna_path  = self.model_path + "vicuna",
            low_resource = False
        )

        self.model.to(self.torch_device)
        self.model.eval()

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        print(f'The model has  {count_parameters(self.model):,} parameters')

    def _generate(self, input):

        segments, sampling_rate, mode = self._prepare_audio_segments(input["audio"], input['task_type'])

        if mode == 'chunked':
            model_predictions = []
            for chunk in segments:
                # if chunk is less than 1 second, pad it to 1 second
                if len(chunk) < sampling_rate:
                    chunk = np.pad(chunk, (0, sampling_rate - len(chunk)), 'constant', constant_values=(0, 0))
                outputs = self.model.generate(audio_array=chunk, sampling_rate=sampling_rate, prompt=input["instruction"], device=self.torch_device)[0]
                model_predictions.append(outputs)
            return ' '.join(model_predictions)

        return self.model.generate(audio_array=segments[0], sampling_rate=sampling_rate, prompt=input["instruction"], device=self.torch_device)[0]
