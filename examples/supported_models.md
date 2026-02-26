
Supported models:


```
# This is a cascaded method with Whisper-large-v3 and LLAMA-3-8B-Instruct
MODEL_NAME=cascade_whisper_large_v3_llama_3_8b_instruct

# This is a cascaded method with Whisper-large-v2 and SEALION-V3 LLM model.
MODEL_NAME=cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct

# The Qwen2-Audio Model: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
MODEL_NAME=qwen2-audio-7b-instruct

# The Qwen-Audio Model: https://huggingface.co/Qwen/Qwen-Audio-Chat
MODEL_NAME=qwen-audio-chat

# This is the SALMONN model: https://arxiv.org/abs/2310.13289
MODEL_NAME=salmonn_7b

# MERaLiON-AudioLLM: https://huggingface.co/MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION
MODEL_NAME=meralion-audiollm-whisper-sea-lion

# Only whisper - for ASR / ST Tasks
MODEL_NAME=whisper_large_v3
MODEL_NAME=whisper_large_v2

# Google Gemini
export GOOGLE_API_KEY=AIzxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MODEL_NAME=gemini-1.5-flash
MODEL_NAME=gemini-2-flash

# OpenAI GPT-4o Audio
MODEL_NAME=gpt-4o-audio

# Microsoft Phi-4 Multimodal
MODEL_NAME=phi_4_multimodal_instruct

# SeaLLMs Audio: https://huggingface.co/SeaLLMs/SeaLLMs-Audio-7B
MODEL_NAME=seallms_audio_7b

# Audio Flamingo
MODEL_NAME=audio_flamingo

# NeMo-based models
MODEL_NAME=canary_qwen
MODEL_NAME=luciole_audio  # prefix match: luciole_audio*

# Qwen2-Omni (prefix match: qwen2_omni*)
MODEL_NAME=qwen2_omni-7b

# Voxtral (prefix match: voxtral*)
MODEL_NAME=voxtral
```


## Preparation for SALMONN_7B

```
# Move to examples folder
cd examples
# need Git LFS to download large model files
# e.g. apt install git-lfs
git clone https://huggingface.co/AudioLLMs/SALMONN_7B

cd ..
bash examples/eval_SALMONN_7B.sh
```


