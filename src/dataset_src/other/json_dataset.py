import logging
import random
import soundfile as sf
from tqdm import tqdm

from dataset_src.base_dataset import BaseDatasetProcessor

class jsonl_dataset_processor(BaseDatasetProcessor):
    
    
    def load(self):
        """Actually load the raw data. Call before prepare_model_input()."""
        raw_data = self._data_loader()
        logging.info(f"Loaded {len(raw_data)} samples")

        if self._min_audio_duration is not None or self._max_audio_duration is not None:
            before = len(raw_data)
            raw_data = [s for s in raw_data if self._check_duration(s)]
            logging.info(f"Duration filter: {before} -> {len(raw_data)} samples "
                         f"(min={self._min_audio_duration}, max={self._max_audio_duration})")

        self._dataset_size = len(raw_data)

        if self._number_of_samples != -1:
            if self._number_of_samples > len(raw_data):
                self._number_of_samples = len(raw_data)
                logging.info(f"Requested samples exceed available. Using {self._number_of_samples}")
            raw_data = raw_data[:self._number_of_samples]

        logging.info(f'Number of samples: {len(raw_data)}')
        input_data = []
        for sample in tqdm(raw_data, desc="Processing samples"):
            input_data.append(self._process_sample(sample))

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        logging.info(random.sample(input_data, 1)[0])
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data
    
    def _check_duration(self, sample):
        dur = self._get_sample_duration(sample)
        if self._min_audio_duration is not None and dur < self._min_audio_duration:
            return False
        if self._max_audio_duration is not None and dur > self._max_audio_duration:
            return False
        return True

    @staticmethod
    def _get_sample_duration(sample):
        conversations = sample["conversations"]
        audio_entry = conversations[1] if conversations[0]["type"] == "text" else conversations[0]
        if "duration" in audio_entry:
            return audio_entry["duration"]
        return sf.info(audio_entry["value"]).duration

    def _process_sample(self, sample):
        """Build one input dict from a raw sample. Override to add extra fields."""
        conversations = sample["conversations"]
        instruction = conversations[0]["value"] if conversations[0]["type"]=="text" else None
        if instruction:
            audio_entry = conversations[1]
            audio = audio_entry["value"]
            reference = conversations[2]["value"]
        else:
            audio_entry = conversations[0]
            audio = audio_entry["value"]
            reference = conversations[1]["value"]
            if self.instructions is not None:
                instruction = random.choice(self.instructions)
            else:
                raise ValueError(f"Missing instructions for sample {sample}")

        # Handle offset/duration for audio segment extraction
        read_kwargs = {}
        if "offset" in audio_entry or "duration" in audio_entry:
            info = sf.info(audio)
            sr = info.samplerate
            if "offset" in audio_entry:
                start = int(audio_entry["offset"] * sr)
                if start >= info.frames:
                    if not self._ignore_offsets:
                        raise ValueError(
                            f"Offset {audio_entry['offset']}s (frame {start}) exceeds "
                            f"file length {info.frames} frames for {audio}. "
                            f"Set ignore_offsets=True in config to read from file start instead."
                        )
                else:
                    read_kwargs["start"] = start
            if "duration" in audio_entry:
                read_kwargs["frames"] = int(audio_entry["duration"] * sr)

        array, sr = sf.read(audio, **read_kwargs)
        if len(array)==0:
            raise ValueError(f"Audio file {audio} is empty.")
        return {
            "audio": dict(array=array, sampling_rate=sr),
            "instruction": instruction,
            "reference": reference,
            "task_type": self.task_type,
            "sub_task": self.sub_task,
            "language": self.language,
        }