<h1 align="center">AudioBench — OpenLLM-France</h1>

<p align="center">
  <a href="https://openllm-france.github.io/AudioBench/"><img src="https://img.shields.io/badge/Leaderboard-live-1f6f78" alt="Leaderboard"></a>
  <a href="https://huggingface.co/datasets/OpenLLM-France/Luciole-Audio-Evaluation-Dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-Luciole--Audio--Evaluation-ff9d00" alt="Evaluation datasets"></a>
  <a href="https://arxiv.org/abs/2406.16020"><img src="https://img.shields.io/badge/arXiv-2406.16020-b31b1b.svg" alt="AudioBench paper"></a>
</p>

Multi-task benchmark of audio language models (ASR, speech translation, spoken QA,
audio/music understanding, speaker attributes…), focused on **French and English**, with
additional results in German, Spanish, Italian, Dutch, Portuguese and Arabic.

This repository is OpenLLM-France's fork of [AudioBench](https://github.com/AudioLLMs/AudioBench)
(Wang et al., NAACL 2025). It is used to evaluate the
[Luciole audio models](https://huggingface.co/OpenLLM-France/Luciole-8B-Audio-1.0) and adds:

- French and multilingual test sets, read from JSONL files (most of them published as the
  [Luciole Audio Evaluation Dataset](https://huggingface.co/datasets/OpenLLM-France/Luciole-Audio-Evaluation-Dataset));
- recent audio LLMs (Qwen2.5/3-Omni, Voxtral, Gemma 4, Granite Speech, Audio Flamingo 3…),
  mostly served with vLLM;
- new metrics (label matching, temporal localization) and
  [Flow-Judge](https://huggingface.co/flowaicom/Flow-Judge-v0.1) as the default LLM judge;
- YAML-driven runs and a static HTML leaderboard.

🏆 **Leaderboard: <https://openllm-france.github.io/AudioBench/>**

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Leaderboard datasets](#leaderboard-datasets)
- [Other supported datasets](#other-supported-datasets)
- [Metrics](#metrics)
- [Supported models](#supported-models)
- [Leaderboard and reports](#leaderboard-and-reports)
- [Adding a dataset or a model](#adding-a-dataset-or-a-model)
- [Citation](#citation)
- [License](#license)

## Installation

```shell
pip install -e ".[vllm,flow-judge,qwen-omni]"   # or ".[all]"
```

Optional extras (see `pyproject.toml`): `vllm` (most models), `flow-judge` (LLM judge),
`qwen-omni`, `api` (Gemini / GPT-4o), `math`, `kimi`, `plot`.
The NeMo-based models (Luciole, Canary-Qwen) also need [NeMo](https://github.com/NVIDIA/NeMo).

## Quick start

Runs are described by a YAML file listing models and datasets:

```shell
export DATA_FOLDER=/path/to/data        # replaces <DATA_FOLDER> in dataset paths
export MODELS_FOLDER=/path/to/models    # replaces <MODELS_FOLDER> in model paths
python -m audio_bench.run_evaluations --config_path configs/example.yaml
```

- `configs/example.yaml`: a minimal example (2 samples per dataset).
- `configs/config.yaml`: the full benchmark used for the leaderboard.

A single (model, dataset) pair can also be run directly:

```shell
python -m audio_bench.main_evaluate --dataset_name librispeech_test_clean \
    --model_name qwen2_audio_7b_instruct --metrics wer --number_of_samples 100
```

For each (model, dataset), the predictions and the scores are written to
`results/<model_id>/<LANG>/<dataset>.json` and `<dataset>_score.json`. Inference is
skipped when the predictions already exist (unless `overwrite: true`), and the scores are
computed again when a metric is missing.

## Configuration

```yaml
global:
  batch_size: 25
  number_of_samples: 250        # -1 = whole test set
  overwrite: false
  output_folder: results
  min_audio_duration: 1         # seconds; filters the samples
  max_audio_duration: 120
  compute_metrics: true         # false: inference only (e.g. score later on another GPU)
  skip_inference: false         # true: only (re)compute the scores
  skip_errors: false
  gpu_memory_utilization: 0.4   # vLLM

  datasets:
    - group: asr_fr             # groups pass their keys down to their datasets
      task: asr
      language: fr
      metrics: wer
      datasets:
        - name: Fleurs
          path: <DATA_FOLDER>/nemo/asr/fr/context/FLEURS/test.jsonl
    - name: librispeech_test_clean   # a built-in dataset (no path)

models:
  - id: qwen2_omni_7b
    gpu_memory_utilization: 0.65
  - id: luciole_audio_8b
    name: OpenLLM-France/Luciole-8B-Audio-1.0   # display name in the results
    path: <MODELS_FOLDER>/luciole-8b-audio
```

Dataset keys: `name`, `path` (JSONL file), `task`, `language`, `sub_task`, `metrics`
(one or a list), `number_of_samples`, `min_audio_duration` / `max_audio_duration`,
`prompt_prefix` (prepended to the instruction), `judge_binary` (binary judge score),
`audio_first`, `ignore_offsets`.
Model keys: `id` (see [Supported models](#supported-models)), `name`, `path`, `batch_size`,
`gpu_memory_utilization`, `device`, `audio_locator_tag`, and `datasets` (overrides the
global list).

**JSONL datasets** use the format of the
[Luciole Audio Evaluation Dataset](https://huggingface.co/datasets/OpenLLM-France/Luciole-Audio-Evaluation-Dataset#data-fields):
one record per line, with `conversations` holding the user instruction, the audio turn
(`value` = audio file path, optional `offset` / `duration`) and the reference answer
(`Assistant` turn). Audio paths are read as written, so relative paths are resolved from
the working directory.

## Leaderboard datasets

The test sets of the leaderboard (`configs/config.yaml`). Most of them are published
(text, and audio when its license allows it) in the
[Luciole Audio Evaluation Dataset](https://huggingface.co/datasets/OpenLLM-France/Luciole-Audio-Evaluation-Dataset),
whose card details each source and its license. The others are loaded from the original
AudioBench datasets on the Hugging Face Hub, or must be obtained from their source.

| Task | Dataset | Languages | Metric |
|---|---|---|---|
| ASR | [FLEURS](https://huggingface.co/datasets/google/fleurs) | fr, en, it, de, nl, es, pt, ar | `wer` |
| ASR | [Common Voice](https://commonvoice.mozilla.org/en/datasets) | fr, en | `wer` |
| ASR | [VoxPopuli](https://github.com/facebookresearch/voxpopuli) | fr, en | `wer` |
| ASR | [Multilingual TEDx](https://www.openslr.org/100/) ¹ | fr, it, de, es, pt, ar | `wer` |
| ASR | [SUMM-RE](https://huggingface.co/datasets/linagora/SUMM-RE) | fr | `wer` |
| ASR | YouTubeFr ² | fr | `wer` |
| Speech translation | [Multilingual TEDx](https://www.openslr.org/100/) ¹ | fr→en, fr→es, es→fr, es→it | `bleu`, `meteor` |
| Spoken QA | VoxPopuli-QA (questions synthesized by OpenLLM-France) | fr | `flow_judge` |
| Spoken QA | [Aya Collection](https://huggingface.co/datasets/CohereLabs/aya_collection) (TTS) | fr | `flow_judge` |
| Spoken QA | [Vigogne Alpaca](https://github.com/bofenghuang/vigogne) (TTS) | fr | `flow_judge` |
| Spoken QA | [SLUE-P2-SQA5](https://huggingface.co/datasets/AudioLLMs/slue_p2_sqa5_test) | en | `flow_judge` |
| Spoken QA | [Public-SG-Speech-QA](https://huggingface.co/datasets/AudioLLMs/public_sg_speech_qa_test) | en | `flow_judge` |
| Spoken QA | [Alpaca-Audio](https://huggingface.co/datasets/AudioLLMs/alpaca_audio_test) | en | `flow_judge` |
| Spoken QA | [OpenHermes-Audio](https://huggingface.co/datasets/AudioLLMs/openhermes_instruction_test) | en | `flow_judge` |
| Spoken QA | [National Speech Corpus SQA](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1) | en | `flow_judge` |
| Spoken QA | Fisher, via [AIR-Bench](https://github.com/OFA-Sys/AIR-Bench) ([LDC](https://catalog.ldc.upenn.edu/LDC2004S13)) ¹ | en | `flow_judge` |
| Spoken QA | [SpokenWOZ](https://spokenwoz.github.io/), via [AIR-Bench](https://github.com/OFA-Sys/AIR-Bench) | en | `flow_judge` |
| Spoken math QA | [Spoken-MQA](https://huggingface.co/datasets/amao0o0/spoken-mqa) (short digit) | en | `acc` |
| Dialogue summarization | [National Speech Corpus SDS](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1) | en | `flow_judge` |
| Temporal localization ³ | [SLUE-SQA5](https://huggingface.co/datasets/asapp/slue-phase-2) (time→word, word→time) | en | `temporal_regex` |
| Audio QA | [Clotho-AQA](https://huggingface.co/datasets/AudioLLMs/clotho_aqa_test), [WavCaps-QA](https://huggingface.co/datasets/AudioLLMs/wavcaps_qa_test), [AudioCaps-QA](https://huggingface.co/datasets/AudioLLMs/audiocaps_qa_test) | en | `flow_judge` |
| Audio captioning | [WavCaps](https://huggingface.co/datasets/AudioLLMs/wavcaps_test), [AudioCaps](https://huggingface.co/datasets/AudioLLMs/audiocaps_test) | en | `flow_judge` |
| Emotion recognition | [MELD](https://affective-meld.github.io/) | fr, en | `label_match` |
| Emotion recognition | [IEMOCAP](https://huggingface.co/datasets/AudioLLMs/iemocap_emotion_recognition) | en | `label_match` |
| Gender recognition | [Common Voice](https://commonvoice.mozilla.org/en/datasets) | fr, en | `label_match` |
| Gender recognition | [IEMOCAP](https://huggingface.co/datasets/AudioLLMs/iemocap_gender_recognition) | en | `label_match` |
| Age recognition | [Common Voice](https://commonvoice.mozilla.org/en/datasets) | fr, en | `label_match` |
| Language identification | [CoVoST 2](https://github.com/facebookresearch/covost), via [AIR-Bench](https://github.com/OFA-Sys/AIR-Bench) | multilingual | `label_match` |
| Diarization | [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) | en | `flow_judge` |
| Music QA | [MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) (Q&A generated by OpenLLM-France) | fr, en | `flow_judge` |
| Music QA | [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset), via [AIR-Bench](https://github.com/OFA-Sys/AIR-Bench) | en | `flow_judge` (binary) |
| Music QA ³ | [MuChoMusic](https://huggingface.co/datasets/AudioLLMs/mu_chomusic_test) | en | `flow_judge` |
| Music captioning | [MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) | en | `flow_judge` |

¹ Not redistributable (license): not in the Luciole Audio Evaluation Dataset, get it from the source.
² Private test set (copyrighted YouTube audio).
³ Hidden by default in the leaderboard (can be ticked back in the dataset filter).

## Other supported datasets

The datasets of the original AudioBench, loaded from the Hugging Face Hub by name (no
`path` needed). Their default metric is used when `metrics` is not set.

| Task | Datasets (`name`) |
|---|---|
| ASR (en) | [`librispeech_test_clean`](https://huggingface.co/datasets/AudioLLMs/librispeech_test_clean), [`librispeech_test_other`](https://huggingface.co/datasets/AudioLLMs/librispeech_test_other), [`common_voice_15_en_test`](https://huggingface.co/datasets/AudioLLMs/common_voice_15_en_test), [`peoples_speech_test`](https://huggingface.co/datasets/AudioLLMs/peoples_speech_test), [`gigaspeech_test`](https://huggingface.co/datasets/AudioLLMs/gigaspeech_test), [`tedlium3_test`](https://huggingface.co/datasets/AudioLLMs/tedlium3_test), [`tedlium3_long_form_test`](https://huggingface.co/datasets/AudioLLMs/tedlium3_long_form_test), [`earnings21_test`](https://huggingface.co/datasets/AudioLLMs/earnings21_test), [`earnings22_test`](https://huggingface.co/datasets/AudioLLMs/earnings22_test) |
| ASR (other languages) | [`aishell_asr_zh_test`](https://huggingface.co/datasets/AudioLLMs/aishell_1_zh_test), [`gigaspeech2_thai`, `gigaspeech2_indo`, `gigaspeech2_viet`](https://huggingface.co/datasets/AudioLLMs/gigaspeech2-test) |
| ASR (code-switching) | [`seame_dev_man`](https://huggingface.co/datasets/AudioLLMs/seame_dev_man), [`seame_dev_sge`](https://huggingface.co/datasets/AudioLLMs/seame_dev_sge) |
| ASR (Singlish) | `imda_part{1..6}_asr_test` ([MNSC](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1)) |
| Speech translation | [`covost2_en_id_test`](https://huggingface.co/datasets/AudioLLMs/covost2_en_id_test), [`covost2_en_zh_test`](https://huggingface.co/datasets/AudioLLMs/covost2_en_zh_test), [`covost2_en_ta_test`](https://huggingface.co/datasets/AudioLLMs/covost2_en_ta_test), [`covost2_id_en_test`](https://huggingface.co/datasets/AudioLLMs/covost2_id_en_test), [`covost2_zh_en_test`](https://huggingface.co/datasets/AudioLLMs/covost2_zh_en_test), [`covost2_ta_en_test`](https://huggingface.co/datasets/AudioLLMs/covost2_ta_en_test) |
| Spoken QA | [`cn_college_listen_mcq_test`](https://huggingface.co/datasets/AudioLLMs/cn_college_listen_mcq_test), [`dream_tts_mcq_test`](https://huggingface.co/datasets/AudioLLMs/dream_tts_mcq_test), [`spoken_squad_test`](https://huggingface.co/datasets/AudioLLMs/spoken_squad_test), `imda_part{3..6}_30s_sqa_human_test` ([MNSC](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1)) |
| Spoken math QA | [`spoken-mqa_long_digit`, `spoken-mqa_single_step_reasoning`, `spoken-mqa_multi_step_reasoning`](https://huggingface.co/datasets/amao0o0/spoken-mqa) |
| Dialogue summarization | `imda_part{3..6}_30s_ds_human_test` ([MNSC](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1)) |
| Paralinguistics | [`voxceleb_accent_test`](https://huggingface.co/datasets/AudioLLMs/voxceleb_accent_test), [`voxceleb_gender_test`](https://huggingface.co/datasets/AudioLLMs/voxceleb_gender_test), [`meld_sentiment_test`](https://huggingface.co/datasets/AudioLLMs/meld_sentiment_test), [`meld_emotion_test`](https://huggingface.co/datasets/AudioLLMs/meld_emotion_test), `imda_ar_sentence`, `imda_ar_dialogue`, `imda_gr_sentence`, `imda_gr_dialogue` ([MNSC](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1)) |
| Audio understanding | [`mmau_mini`](https://huggingface.co/datasets/AudioLLMs/MMAU-mini) |
| Instruction following | [`audiollm_instructionfollowing`](https://huggingface.co/datasets/YichenG170/AudioLLMInstructionFollowing) |

## Metrics

| Metric | Description |
|---|---|
| `wer` | Word error rate, after text normalization (numbers written as words); capped at 100% per segment |
| `bleu`, `meteor` | Speech translation and captioning |
| `flow_judge` | LLM judge ([Flow-Judge v0.1](https://huggingface.co/flowaicom/Flow-Judge-v0.1), run locally with vLLM), 1–5 score or binary (`judge_binary`) |
| `label_match` | Classification: the first known label found in the answer |
| `temporal_regex` | Temporal localization (word ↔ timestamp) |
| `acc`, `string_match` | Exact answer (math QA, multiple choice) |
| `flow_judge_api`, `gpt4o_judge`, `llama3_70b_judge`, `linagora_api_oss120` | Other judges, served through an API |

## Supported models

The `id` to use in the configs. The backend is fixed by the model.

| `id` | Model | Backend |
|---|---|---|
| `luciole_audio*`, `linagora*` | [OpenLLM-France/Luciole-8B-Audio-1.0](https://huggingface.co/OpenLLM-France/Luciole-8B-Audio-1.0) (set `path`) | NeMo |
| `canary_qwen` | [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) | NeMo |
| `qwen2_audio_7b_instruct` | [Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | vLLM |
| `qwen2_omni_7b`, `qwen2_omni_3b` | [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | vLLM |
| `qwen3_omni_30b_instruct` | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | vLLM |
| `voxtral_mini_3b_2507`, `voxtral_small_24b_2507` | [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), [mistralai/Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) | vLLM |
| `gemma_4_e2b_it`, `gemma_4_e4b_it` | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it), [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | vLLM |
| `granite_speech_4_1_2b` | [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | vLLM |
| `phi_4_multimodal_instruct` | [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | vLLM |
| `audio_flamingo_3`, `audio_flamingo_next` | [nvidia/audio-flamingo-3-hf](https://huggingface.co/nvidia/audio-flamingo-3-hf), [nvidia/audio-flamingo-next-hf](https://huggingface.co/nvidia/audio-flamingo-next-hf) | vLLM |
| `seallms_audio_7b` | [SeaLLMs/SeaLLMs-Audio-7B](https://huggingface.co/SeaLLMs/SeaLLMs-Audio-7B) | vLLM |
| `whisper_large_v3`, `whisper_large_v2` | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) | vLLM |
| `kimi_audio_7b_instruct` | [moonshotai/Kimi-Audio-7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct) | Transformers |
| `qwen_audio_chat` | [Qwen/Qwen-Audio-Chat](https://huggingface.co/Qwen/Qwen-Audio-Chat) | Transformers |
| `salmonn_7b` | [tsinghua-ee/SALMONN-7B](https://huggingface.co/tsinghua-ee/SALMONN-7B) (needs an extra git clone) | Transformers |
| `meralion_audiollm_whisper_sea_lion` | [MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION](https://huggingface.co/MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION) | Transformers |
| `cascade_whisper_large_v3_llama_3_8b_instruct` | [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) + [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Transformers |
| `cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct` | [Whisper large-v2](https://huggingface.co/openai/whisper-large-v2) + [Gemma2-9B-CPT-SEA-LIONv3](https://huggingface.co/aisingapore/gemma2-9b-cpt-sea-lionv3-instruct) | Transformers |
| `wavllm_fairseq` | [WavLLM](https://github.com/microsoft/SpeechT5/tree/main/WavLLM) (no longer maintained) | fairseq |
| `gemini_1_5_flash`, `gemini_2_flash` | [Gemini](https://ai.google.dev/gemini-api/docs/models) (API key needed) | API |
| `gpt_4o_audio` | [GPT-4o audio](https://platform.openai.com/docs/models/gpt-4o-audio-preview) (Azure OpenAI key needed) | API |

## Leaderboard and reports

The [leaderboard](https://openllm-france.github.io/AudioBench/) is a static page built from
the score files in `leaderboard/results/` and from `leaderboard/config.yaml` (shown
datasets and models, model sizes, Hugging Face links). It is rebuilt and deployed to GitHub
Pages by `.github/workflows/leaderboard.yml` on every push to `main` that changes
`leaderboard/` or `audio_bench/visualization/`. To build it locally:

```shell
python -m audio_bench.visualization.build_leaderboard leaderboard/results \
    --config leaderboard/config.yaml --output_folder leaderboard
# then open leaderboard/index.html
```

The same command on a `results/` folder gives a report of your own runs
(`--show_all_models`, `--show_all_datasets` to bypass the config filters).
Other tools:

- `audio_bench.visualization.plot_radar`, `audio_bench.visualization.plot_category_bars`: static figures;
- `audio_bench.compare_two_systems`: significance tests between two models;
- `tools/compare_results.py`: side-by-side predictions of several models on a dataset;
- `audio_bench.table_to_png`: exports a table of the report as PNG.

## Adding a dataset or a model

- **JSONL dataset**: no code needed, add an entry with a `path`, `task`, `language` and
  `metrics` to your config (see [Configuration](#configuration)).
- **Hugging Face dataset**: add it to `DATASET_SOURCES` and `_create_processor` in
  `audio_bench/dataset_factory.py`, with a processor in `audio_bench/dataset_src/`.
- **Model**: subclass `VLLMModel`, `NeMoModel` or `BaseModel` (`audio_bench/model_src/`),
  register it in `load_model` and `_MODEL_ID_TO_NAME` in `audio_bench/model_factory.py`.

## Citation

If you use this benchmark, please cite the [OpenLLM-France](https://huggingface.co/OpenLLM-France) /
Luciole project and the individual source datasets you rely on (see the
[Luciole Audio Evaluation Dataset card](https://huggingface.co/datasets/OpenLLM-France/Luciole-Audio-Evaluation-Dataset#datasets-and-licenses)).

This benchmark builds on AudioBench; please cite it too:

```bibtex
@article{wang2024audiobench,
  title={AudioBench: A Universal Benchmark for Audio Large Language Models},
  author={Wang, Bin and Zou, Xunlong and Lin, Geyu and Sun, Shuo and Liu, Zhuohan and Zhang, Wenyu and Liu, Zhengyuan and Aw, AiTi and Chen, Nancy F},
  journal={NAACL},
  year={2025}
}
```

## License

The code is under a Creative Commons NonCommercial license (see [LICENSE](LICENSE)). Each
dataset remains under its own license.
