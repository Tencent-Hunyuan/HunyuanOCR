<div align="center">

[中文阅读](./README_zh.md)

# HunyuanOCR-1.5: Making Lightweight OCR VLMs Faster and Better <!-- omit in toc -->

<img src="./assets/HyOCR_1_5_teaser.png" width="90%"/> <br>

<a href="https://huggingface.co/tencent/HunyuanOCR"><b>🤗 HF Model</b></a> |
<a href="https://arxiv.org/pdf/2607.04884"><b>📄 Paper</b></a>

</div>

> [!NOTE]
> 👉 Looking for the original **HunyuanOCR-1.0** release? Switch to the [`v1.0`](https://github.com/Tencent-Hunyuan/HunyuanOCR/tree/v1.0) branch, or read [`README_v1.0.md`](./HunyuanOCR_v1.0/README_v1.0.md) · [`README_zh_v1.0.md`](./HunyuanOCR_v1.0/README_zh_v1.0.md)。

---

## 🔥 News <!-- omit in toc -->

- **[2026/07/24]** 🚀 We further open-sourced the **unified inference environment** (see [`docs/inference/inference.md`](docs/inference/inference.md)) and the **verl-based reinforcement learning stack** for HunyuanOCR-1.5 (see [`train_verl/README_en.md`](train_verl/README_en.md)).
- **[2026/07/13]** 📊 We open-sourced [**CHAOS-Bench**](./benchmarks/CHAOS-Bench), a character-level hallucination benchmark that probes the "seeing-is-believing" ability of OCR VLMs by injecting character-level corruptions into academic-paper images.
- **[2026/07/07]** 🚀 We released **HunyuanOCR-1.5**, a systematic upgrade that makes lightweight end-to-end OCR **faster and better** via DFlash speculative decoding, PC-side llama.cpp deployment, an Agentic Data Flow, and an upgraded training recipe. Check out the [paper](https://arxiv.org/pdf/2607.04884).
- **[2026/06/18]** 🎉 Our work on table parsing has been accepted to the ECCV 2026 Main Conference! Check out the paper: [StrucTab: A Structured Optimization Framework for Table Parsing](https://arxiv.org/abs/2606.29905).
- **[2026/06/02]** 🎉 We have released two new benchmarks. [Chronicles-OCR](https://github.com/VirtualLUOUCAS/Chronicles-OCR) ([arXiv](https://arxiv.org/abs/2605.11960)), an open-source ancient-text perception benchmark covering the evolutionary trajectory of the "Seven Chinese Scripts", is jointly built by the **SSV Digital Culture Lab** and the **SSV Technical Architecture Department**, together with the **Palace Museum** and **Anyang Normal University**. We have also released [ChartArena](https://github.com/pspdada/ChartArena) ([arXiv](https://arxiv.org/abs/2606.01348)), a new chart-parsing benchmark supporting diverse chart types. Welcome to evaluate and provide your valuable feedback!
- **[2026/05/11]** 🎉 We have officially open-sourced two benchmarks on document parsing and text-image machine translation: [Wild-OmniDocBench](https://github.com/VirtualLUOUCAS/Wild_OmniDocBench) and [MMTIT-Bench](https://github.com/VirtualLUOUCAS/MMTIT_Bench). Welcome to evaluate and provide your valuable feedback!
- **[2026/04/08]** 🎉 Our works on document parsing and text-image machine translation have been accepted to the CVPR 2026 Main Conference! Check out the papers: [Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](https://arxiv.org/abs/2603.23885) and [MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](https://arxiv.org/abs/2603.23896).

<details>
<summary>📜 Archived news (HunyuanOCR-1.0)</summary>

- **[2026/01/13]** ⭐ We have released a stable official [online demo](https://hunyuan.tencent.com/chat/HunyuanDefault?modelId=HY-OCR-1.0&mid=308&from=vision-zh), feel free to try it out!
- **[2025/11/28]** 🛠️ We fixed vLLM inference bugs and hyperparameter configuration issues such as system prompt. It is recommended to use the latest vLLM installation steps and the [inference script](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/main/Hunyuan-OCR-master/Hunyuan-OCR-vllm/run_hy_ocr.py) for performance testing. Currently, there is still a certain accuracy difference between Transformers and the vLLM framework (we are working on fixing this).
- **[2025/11/25]** 📝 Inference code and model weights publicly available.

</details>

---

## 📌 Contents <!-- omit in toc -->

- [📖 Introduction](#-introduction)
- [⚙️ Environment](#️-environment)
  - [Training](#training)
  - [Inference](#inference)
- [🚀 Training](#-training)
  - [1. Prepare packed training data](#1-prepare-packed-training-data)
  - [2. SFT the HunyuanOCR base model](#2-sft-the-hunyuanocr-base-model)
  - [3. Train the DFlash draft model — from scratch](#3-train-the-dflash-draft-model--from-scratch)
  - [4. Continue-finetune from an existing DFlash checkpoint](#4-continue-finetune-from-an-existing-dflash-checkpoint)
- [🧪 Inference](#-inference)
  - [Environment setup](#environment-setup)
  - [Download the weights](#download-the-weights)
  - [Quick start (vLLM AR, single GPU)](#quick-start-vllm-ar-single-gpu)
  - [PC-side deployment via llama.cpp](#pc-side-deployment-via-llamacpp)
- [📖 Documentation](#-documentation)
- [📚 Citation](#-citation)

## 📖 Introduction

**HunyuanOCR-1.5** is a lightweight, end-to-end OCR-specialized vision-language model. It targets a broad range of text-centric visual tasks and unifies **document parsing, text spotting, information extraction, text-image translation** within a single end-to-end VLM.

Building upon the validated lightweight architecture of HunyuanOCR-1.0, HunyuanOCR-1.5 does **not** redesign the model backbone. Instead, it performs a systematic upgrade around two goals — **making the model faster and better**:

- ⚡ **Faster — DFlash inference acceleration.**
  End-to-end OCR is often accompanied by long autoregressive decoding, which becomes the major
  bottleneck for dense documents, tables, formulas, and other long structured outputs.
  HunyuanOCR-1.5 adapts a speculative-decoding framework based on **DFlash**: a lightweight
  block-diffusion draft model drafts multiple candidate tokens in parallel, which are then verified
  by the target model in a single pass. This significantly reduces the decoding latency of long
  structured outputs while **preserving the output distribution** of the target model.

- 💻 **PC-side deployment via llama.cpp.**
  Beyond server-grade vLLM, HunyuanOCR-1.5 also supports **CPU / consumer-GPU / laptop** deployment
  through [`llama.cpp`](https://github.com/ggml-org/llama.cpp) with a GGUF-converted checkpoint and
  an OpenAI-compatible `llama-server`. A DFlash-adapted `llama.cpp` fork is provided as well, so the
  same speculative-decoding acceleration is available on PC. See
  [`docs/llama_cpp.md`](docs/llama_cpp.md).

- 🧠 **Better — Agentic Data Flow + upgraded training recipe.**
  On the data side, we propose **Agentic Data Flow**, an agent-driven data-construction system that
  translates model weaknesses into executable data requirements. Agents deeply participate in
  material search, tool-based verification, sample cleaning, and data-pipeline development, and
  iterate in a closed loop with algorithm engineers. In HunyuanOCR-1.5, this system is used for
  targeted long-tail capabilities such as **low-resource OCR, ancient-script OCR, and multi-image
  text-centric QA**.
  On the training side, we systematically upgrade the recipe: pretraining Stage-3 is re-planned to
  incorporate the newly produced capability data, multi-image data, and historical OCR data, with
  maximum image resolution extended to **4K** and context window extended to **128K**; post-training
  refines the SFT data and further explores RL across different OCR tasks to amplify the gains from
  reinforcement learning.

Together, HunyuanOCR-1.5 achieves **both faster inference and broader OCR capability coverage**
while retaining the deployment advantages of a lightweight end-to-end model. This repository
open-sources the SFT / DFlash training pipeline and the transformers / vLLM inference stack, so
that the community can reproduce, fine-tune, and extend OCR-specialized VLMs.

---

## ⚙️ Environment

### Training

- Python 3.10+ (3.12 tested)
- PyTorch 2.1+ (CUDA 12.1+; a cu130 build has been tested end-to-end)
- transformers 4.57+
- DeepSpeed 0.14+

```bash
pip install -r requirements.txt
# flash-attn requires a manual build:
pip install flash-attn --no-build-isolation
```

### Inference

Inference now uses a **single unified environment** (built on `uv`, requires
CUDA 13) that runs all three configurations from the same install: **vLLM AR,
DFlash speculative decoding, and native transformers**. Accuracy alignment
across the three has been verified.

```bash
pip install uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "vllm>=0.25.1"
uv pip install --no-build-isolation --no-cache-dir "flash-attn==2.8.3"
```

The inference code lives under [`inference/`](inference)
(`inference/vLLM`, `inference/DFlash`, `inference/transformers`). See
[`docs/inference/inference.md`](docs/inference/inference.md) for the full setup
and usage. If you lack CUDA 13 or only need one configuration, that document
also points to the lighter per-configuration recipes.

---

## 🚀 Training

All training scripts live under `scripts/` and share `scripts/env_common.sh` for distributed env variables. Multi-node training is supported via the standard
`NNODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` env vars.

### 1. Prepare packed training data

We tokenize each raw OCR JSONL, then pack multiple samples up to `packed_max_length=20480`
tokens into single sequences to maximize GPU utilization.

**Step 1** — fill in `configs/data_list.txt` with one absolute path per line, each pointing to a
raw OCR JSONL file. The JSONL schema is documented in [`docs/data_format.md`](docs/data_format.md).

**Step 2** — run the multi-process count-and-pack pipeline:

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
INPUT_LIST=./configs/data_list.txt \
PACK_LEN=20480 \
NUM_PROCESSES=32 \
THREADS_PER_PROCESS=8 \
bash scripts/pack_data.sh
```

Output: `./data/parsing_packed_20480.jsonl` — a single sequence-packed JSONL ready for training.

The pipeline is implemented in [`tools/pipeline_count_and_pack.py`](tools/pipeline_count_and_pack.py)
and [`tools/pack_from_counted.py`](tools/pack_from_counted.py).

### 2. SFT the HunyuanOCR base model

Full end-to-end SFT (vision encoder + MLP + LLM) on packed OCR sequences.
Default profile: `lr=2e-5`, `epochs=5`, per-GPU batch=1, `packed_max_length=20480`.

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
NPROC_PER_NODE=8 \
bash scripts/sft_base.sh
```

Entry: [`train/train_hunyuan.py`](train/train_hunyuan.py).
Full argument list: see [`docs/training.md`](docs/training.md).

### 3. Train the DFlash draft model — from scratch

Trains a small block-diffusion draft that predicts K speculative tokens for HunyuanOCR.
Default profile: `lr=1e-4`, `epochs=2`, `num_mask_tokens=16`, `sample_block_num=8`.

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
NPROC_PER_NODE=8 \
bash scripts/sft_dflash.sh
```

Entry: [`train/train_draft.py`](train/train_draft.py).

### 4. Continue-finetune from an existing DFlash checkpoint

Use this when adapting a released DFlash draft to a smaller / domain-specific dataset.
Recommended profile: `lr=2e-5`, `epochs=10`, `warmup_ratio=0.05`, `save_steps=500`.

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
DFLASH_INIT=/path/to/existing/dflash/checkpoint \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
NPROC_PER_NODE=8 \
bash scripts/sft_dflash_finetune.sh
```

Entry: [`train/train_draft_from_dflash.py`](train/train_draft_from_dflash.py).

---

## 🧪 Inference

HunyuanOCR-1.5 uses a **single unified environment** under [`inference/`](inference)
that runs three server/inference configurations — **vLLM AR**, **DFlash
speculative decoding**, and **native transformers** — plus an optional PC-side
path via llama.cpp. All configurations share the same weights and the same
task-type prompts + sampling + post-processing, so their outputs are directly
comparable, and we have verified their accuracy alignment.

- **vLLM AR** — autoregressive serving. → [`inference/vLLM`](inference/vLLM)
- **DFlash** — AR + DFlash speculative decoding (lossless acceleration for long
  outputs). → [`inference/DFlash`](inference/DFlash)
- **transformers** — native multi-GPU HuggingFace inference (alignment / accuracy
  checks). → [`inference/transformers`](inference/transformers)
- **llama.cpp** — CPU / consumer-GPU / laptop, GGUF deployment (see below).

> The unified environment requires **CUDA 13**. If you lack CUDA 13 or only need
> one configuration, see [`docs/inference/inference.md`](docs/inference/inference.md) for the lighter
> per-configuration recipes.

### Environment setup

```bash
pip install uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "vllm>=0.25.1"
uv pip install --no-build-isolation --no-cache-dir "flash-attn==2.8.3"
```

### Download the weights

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

The download contains both the base model and the `dflash/` draft model.

### Quick start (vLLM AR, single GPU)

Launch the OpenAI-compatible server (served as `tencent/HunyuanOCR`, `-tp 1`,
`--max-model-len 131072`):

```bash
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/vLLM/serve.sh
curl -sf http://127.0.0.1:8000/v1/models     # readiness check
```

Send a single image. The prompt is locked to an official task type via
`--task-type` (run `--list-tasks` to see all 12); sampling
(`temperature=0.0`, `top_p=1.0`, `top_k=-1`, `repetition_penalty=1.08`) and
tail-repetition early-stop / cleanup are built in:

```bash
python inference/vLLM/infer_vllm_client.py \
    --image /path/to/document.png --task-type doc_parse \
    --model tencent/HunyuanOCR --port 8000 --max-tokens 32768
```

Batch inference over a directory (multi-endpoint concurrency, resumable):

```bash
python inference/vLLM/batch_infer.py \
    --image-dir /path/to/images --out-dir /path/to/output \
    --ports 8000 --task-type doc_parse --max-tokens 32768 --concurrency 16
```

For **DFlash acceleration**, use [`inference/DFlash`](inference/DFlash)
(`serve_DFlash.sh`); for **native transformers**, use
[`inference/transformers`](inference/transformers). See
[`docs/inference/inference.md`](docs/inference/inference.md) for the full setup, the task-type table,
and multi-GPU instructions.

### PC-side deployment via llama.cpp

For **CPU / consumer-GPU / laptop** environments, HunyuanOCR-1.5 can also be deployed through
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) after converting the checkpoint to GGUF.
Both the community `llama.cpp` (HunyuanOCR base only) and a DFlash-adapted fork
([`wendadawen/llama.cpp @ dflash-adapt-hunyuanocr-hunyuanstyle`](https://github.com/wendadawen/llama.cpp/tree/dflash-adapt-hunyuanocr-hunyuanstyle))
are supported.

Minimal build & serve (community, no DFlash):

```bash
# 1. Build
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON     # add -DGGML_CUDA=ON for NVIDIA GPU
cmake --build ./build --config Release -j

# 2. Convert HunyuanOCR to GGUF (base + mmproj)
hf download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
python3 convert_hf_to_gguf.py --outfile ./HunyuanOCR/hyocr-f16.gguf        --outtype f16 ./HunyuanOCR
python3 convert_hf_to_gguf.py --outfile ./HunyuanOCR/mmproj-hyocr-f16.gguf --outtype f16 --mmproj ./HunyuanOCR

# 3. Serve (OpenAI-compatible)
build/bin/llama-server \
    --model  ./HunyuanOCR/hyocr-f16.gguf \
    --mmproj ./HunyuanOCR/mmproj-hyocr-f16.gguf \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096
```

DFlash-adapted variant, weight conversion for the draft, and a smoke-test client
([`llama_cpp/chat.py`](llama_cpp/chat.py) with 26 sample OCR images under
[`llama_cpp/test_assets/`](llama_cpp/test_assets)):

see [`docs/llama_cpp.md`](docs/llama_cpp.md) for the complete guide.

---

## 📖 Documentation

- [`docs/training.md`](docs/training.md) — training modes, hyperparameters, distributed setup
- [`docs/data_format.md`](docs/data_format.md) — raw OCR JSONL schema and packing pipeline
- [`docs/inference/inference.md`](docs/inference/inference.md) — unified inference environment (vLLM AR / DFlash / transformers) + deployment tuning
- [`docs/llama_cpp.md`](docs/llama_cpp.md) — PC-side deployment with llama.cpp (community & DFlash-adapted fork)
- [`docs/benchmark.md`](docs/benchmark.md) — end-to-end speed benchmark

---

## 📚 Citation

```bibtex
@article{HunyuanOCR_1_5_2026,
  title   = {{HunyuanOCR-1.5}: Making Lightweight {OCR} {VLMs} Faster and Better},
  author  = {Li, Gengluo and Wan, Xingyu and Peng, Shangpin and Wang, Weinong and Feng, Hao and Du, Yongkun and Wu, Binghong and Ruan, Zheng and Lu, Zhiqiong and Wu, Liang and Lyu, Pengyuan and Shen, Huawen and Lin, Zibin and Hu, Shijing and Yang, Jieneng and Wen, Hongbing and Yu, Guanghua and Liu, Hong and Wang, Bochao and Ma, Can and Hu, Han and Zhang, Chengquan and Zhou, Yu},
  journal = {arXiv preprint arXiv:2607.04884},
  year    = {2026}
}

@article{HunyuanOCR_2025,
  title   = {{HunyuanOCR Technical Report}},
  author  = {Team, Hunyuan Vision and Lyu, Pengyuan and Wan, Xingyu and Li, Gengluo and Peng, Shangpin and Wang, Weinong and Wu, Liang and Shen, Huawen and Zhou, Yu and Tang, Canhui and Yang, Qi and Peng, Qiming and Luo, Bin and Yang, Hower and Zhang, Xinsong and Zhang, Jinnian and Peng, Houwen and Yang, Hongming and Xie, Senhao and Zhou, Longsha and Pei, Ge and Wu, Binghong and Yan, Rui and Wu, Kan and Yang, Jieneng and Wang, Bochao and Liu, Kai and Zhu, Jianchen and Jiang, Jie and Linus and Hu, Han and Zhang, Chengquan},
  journal = {arXiv preprint arXiv:2511.19575},
  year    = {2025}
}

@inproceedings{MMTIT_Bench_2026,
  title     = {{MMTIT-Bench}: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation},
  author    = {Li, Gengluo and Zhang, Chengquan and Liang, Yupu and Shen, Huawen and Zhang, Yaping and Lyu, Pengyuan and Wang, Weinong and Wan, Xingyu and Zeng, Gangyan and Hu, Han and others},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages     = {16593--16602},
  year      = {2026}
}

@article{li2026towardsrealworlddocument,
  title   = {Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training},
  author  = {Li, Gengluo and Lyu, Pengyuan and Zhang, Chengquan and Shen, Huawen and Wu, Liang and Wan, Xingyu and Zeng, Gangyan and Hu, Han and Ma, Can and Zhou, Yu},
  journal = {arXiv preprint arXiv:2603.23885},
  year    = {2026}
}

@article{Chronicles_OCR_2026,
  title   = {{Chronicles-OCR}: A Cross-Temporal Perception Benchmark for the Evolutionary Trajectory of Chinese Characters},
  author  = {Li, Gengluo and Peng, Shangpin and Wan, Xingyu and Zhang, Chengquan and Feng, Hao and Xu, Xin and Wu, Pian and Li, Bang and Ding, Zengmao and Liu, Yongge and others},
  journal = {arXiv preprint arXiv:2605.11960},
  year    = {2026}
}

@article{ChartArena_2026,
  title   = {{ChartArena}: Benchmarking Chart Parsing across Languages, Scenarios, and Formats},
  author  = {Peng, Shangpin and Li, Gengluo and Wan, Xingyu and Zhang, Chengquan and Feng, Hao and Wu, Binghong and Shen, Huawen and Wang, Weinong and Cai, Ziyi and Tian, Zhuotao and Hu, Han and Ma, Can and Zhou, Yu},
  journal = {arXiv preprint arXiv:2606.01348},
  year    = {2026}
}

@article{StrucTab_2026,
  title   = {{StrucTab}: A Structured Optimization Framework for Table Parsing},
  author  = {Li, Gengluo and Peng, Shangpin and Zhang, Chengquan and Wu, Binghong and Feng, Hao and Wang, Weinong and Lyu, Pengyuan and Shen, Huawen and Wan, Xingyu and Tian, Zhuotao and Hu, Han and Ma, Can and Zhou, Yu},
  journal = {arXiv preprint arXiv:2606.29905},
  year    = {2026}
}
```

---

## 🙏 Acknowledgements <!-- omit in toc -->

We would like to thank [Qwen](https://github.com/QwenLM/Qwen3.6) and [DFlash](https://github.com/z-lab/dflash) for their valuable models and ideas.

Special thanks to the Hugging Face community for their Day-0 support.

---

## 📜 License <!-- omit in toc -->

HunyuanOCR-1.5 is released under the same license as HunyuanOCR-1.0 —
the **Tencent Hunyuan Community License Agreement**. See [`LICENSE`](LICENSE) for the full terms.
