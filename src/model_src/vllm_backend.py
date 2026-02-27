import base64
import io
import soundfile as sf


def _audio_to_base64_wav(audio_array, sampling_rate):
    """Convert a numpy audio array to a base64-encoded WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio_array, sampling_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _input_audio_part(audio_array, sampling_rate):
    """Build an OpenAI-style 'input_audio' content part from a numpy array."""
    return {
        "type": "input_audio",
        "input_audio": {
            "data": _audio_to_base64_wav(audio_array, sampling_rate),
            "format": "wav",
        },
    }


def _whisper_task_prompt(task_type, language=None):
    if task_type == 'ASR' and language == 'ZH':
        return "<|startoftranscript|><|zh|><|transcribe|>"
    elif task_type == 'ASR':
        return "<|startoftranscript|>"
    elif task_type == 'AST':
        return "<|startoftranscript|><|en|><|translate|>"
    else:
        raise NotImplementedError(f"Whisper does not support other task: {task_type}.")
