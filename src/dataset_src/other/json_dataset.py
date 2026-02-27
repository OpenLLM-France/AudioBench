import logging
import random
import soundfile as sf

from dataset_src.base_dataset import BaseDatasetProcessor

class jsonl_dataset_processor(BaseDatasetProcessor):
    
    
    def load(self):
        """Actually load the raw data. Call before prepare_model_input()."""
        raw_data = self._data_loader()
        logging.info(f"Loaded {len(raw_data)} samples")

        if self._number_of_samples != -1:
            if self._number_of_samples > len(raw_data):
                self._number_of_samples = len(raw_data)
                logging.info(f"Requested samples exceed available. Using {self._number_of_samples}")
            raw_data = raw_data[:self._number_of_samples]

        self.raw_data = raw_data
        logging.info(f'Number of samples: {len(self.raw_data)}')
        return self
    
    def _process_sample(self, sample):
        """Build one input dict from a raw sample. Override to add extra fields."""
        conversations = sample["conversations"]
        instruction = conversations[0]["value"] if conversations[0]["type"]=="text" else None
        if instruction:
            audio = conversations[1]["value"]
            reference = conversations[2]["value"]
        else:
            audio = conversations[0]["value"]
            reference = conversations[1]["value"]
            if self.instructions is not None:
                instruction = random.choice(self.instructions)
            else:
                raise ValueError(f"Missing instructions for sample {sample}")
        array, sr = sf.read(audio)
        return {
            "audio": dict(array=array, sampling_rate=sr),
            "instruction": instruction,
            "reference": reference,
            "task_type": self.task_type,
            "sub_task": self.sub_task,
            "language": self.language,
        }