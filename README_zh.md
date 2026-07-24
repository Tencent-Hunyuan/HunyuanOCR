<div align="center">

[English Version](./README.md)

# HunyuanOCR-1.5: Making Lightweight OCR VLMs Faster and Better <!-- omit in toc -->

<img src="./assets/HyOCR_1_5_teaser.png" width="90%"/> <br>

<a href="https://huggingface.co/tencent/HunyuanOCR"><b>🤗 HF 模型</b></a> |
<a href="https://arxiv.org/pdf/2607.04884"><b>📄 论文</b></a>

</div>

> [!NOTE]
> 👉 需要原始的 **HunyuanOCR-1.0** 版本？请切换到 [`v1.0`](https://github.com/Tencent-Hunyuan/HunyuanOCR/tree/v1.0) 分支，或阅读 [`README_v1.0.md`](./HunyuanOCR_v1.0/README_v1.0.md) · [`README_zh_v1.0.md`](./HunyuanOCR_v1.0/README_zh_v1.0.md)。

---

## 🔥 最新动态 <!-- omit in toc -->

- **[2026/07/24]** 🚀 我们进一步开源了**统一推理环境**（参考 [`docs/inference/inference_zh.md`](docs/inference/inference_zh.md)），以及基于 verl 的**强化学习训练配套**，让 HunyuanOCR-1.5 的 RL 训练开箱可用（参考 [`train_verl/README.md`](train_verl/README.md)）。
- **[2026/07/13]** 📊 我们开源了 [**CHAOS-Bench**](./benchmarks/CHAOS-Bench)，一个字符级幻觉评测基准：通过在学术论文图像中注入字符级篡改，检验 OCR VLM 的"所见即所得"能力。
- **[2026/07/07]** 🚀 我们发布 **HunyuanOCR-1.5**，通过 DFlash 投机解码、llama.cpp PC 端部署、Agentic Data Flow 及优化后的训练配方，对轻量级端到端 OCR 进行系统性升级，实现**更快、更强**。详见[论文](https://arxiv.org/pdf/2607.04884)。
- **[2026/06/18]** 🎉 我们在表格解析方向的研究成果被 ECCV 2026 Main Conference 正式接收！详见论文：[StrucTab: A Structured Optimization Framework for Table Parsing](https://arxiv.org/abs/2606.29905)。
- **[2026/06/02]** 🎉 我们发布了两项全新评测基准。[Chronicles-OCR](https://github.com/VirtualLUOUCAS/Chronicles-OCR)（[arXiv](https://arxiv.org/abs/2605.11960)）是涵盖"汉字七体"演变轨迹的古文感知开源评测集，由 **SSV 数字文化实验室**、**SSV 技术架构部**联合**故宫博物院**与**安阳师范学院**共同打造；同时发布 [ChartArena](https://github.com/pspdada/ChartArena)（[arXiv](https://arxiv.org/abs/2606.01348)），支持多种图表类型的图表解析评测基准。欢迎大家评测使用并提出宝贵意见！
- **[2026/05/11]** 🎉 我们在文档解析（[Wild-OmniDocBench](https://github.com/VirtualLUOUCAS/Wild_OmniDocBench)）与文本图像机器翻译（[MMTIT-Bench](https://github.com/VirtualLUOUCAS/MMTIT_Bench)）方向的两项 Benchmark 已正式开源，欢迎大家评测使用并提出宝贵意见！
- **[2026/04/08]** 🎉 我们在文档解析与文本图像机器翻译方向的两项研究成果被 CVPR 2026 Main Conference 正式接收！详见论文：[2603.23885](https://arxiv.org/abs/2603.23885)、[2603.23896](https://arxiv.org/abs/2603.23896)。

<details>
<summary>📜 历史归档动态（HunyuanOCR-1.0）</summary>

- **[2026/01/13]** ⭐ 我们发布了稳定的官方[在线 Demo](https://hunyuan.tencent.com/chat/HunyuanDefault?modelId=HY-OCR-1.0&mid=308&from=vision-zh) 页面, 欢迎试用！
- **[2025/11/28]** 🛠️ 我们修复了 vLLM 推理 bug 以及 system prompt 等超参配置问题。建议使用最新的 vLLM 安装步骤和[推理脚本](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/main/Hunyuan-OCR-master/Hunyuan-OCR-vllm/run_hy_ocr.py)进行效果测试。目前 Transformers 相比 vLLM 框架仍然存在一定的精度差异（正在努力修复中）。
- **[2025/11/25]** 📝 推理代码和模型权重已开源。

</details>

---

## 📌 目录 <!-- omit in toc -->

- [📖 简介](#-简介)
- [⚙️ 环境](#️-环境)
  - [训练](#训练)
  - [推理](#推理)
- [🚀 训练](#-训练)
  - [1. 准备打包（packed）后的训练数据](#1-准备打包packed后的训练数据)
  - [2. 对 HunyuanOCR 基座模型进行 SFT](#2-对-hunyuanocr-基座模型进行-sft)
  - [3. 从零训练 DFlash 草稿模型](#3-从零训练-dflash-草稿模型)
  - [4. 从已有 DFlash 检查点继续微调](#4-从已有-dflash-检查点继续微调)
- [🧪 推理](#-推理)
  - [环境安装](#环境安装)
  - [下载权重](#下载权重)
  - [快速开始（vLLM AR，单卡）](#快速开始vllm-ar单卡)
  - [PC 端部署（llama.cpp）](#pc-端部署llamacpp)
- [📖 文档](#-文档)
- [📚 引用](#-引用)

## 📖 简介

**HunyuanOCR-1.5** 是一款轻量化的端到端 OCR 专用视觉语言模型（VLM）。它面向广泛的以文字为中心的视觉任务，将**文档解析、文字检测识别（Text Spotting）、信息抽取、图文翻译**统一到单个端到端 VLM 中。

在延续 HunyuanOCR-1.0 已验证的轻量化架构基础上，HunyuanOCR-1.5 **并未**重新设计模型主干，而是围绕**"更快、更强"**两个目标进行系统性升级：

- ⚡ **更快：DFlash 推理加速。**
  端到端 OCR 通常伴随较长的自回归解码，这在稠密文档、表格、公式等长结构化输出场景中会成为主要瓶颈。HunyuanOCR-1.5 适配了基于 **DFlash** 的投机解码（speculative decoding）框架：一个轻量的块扩散（block-diffusion）草稿模型并行起草多个候选 token，再由目标模型一次性验证。这显著降低了长结构化输出的解码延迟，同时**保持目标模型的输出分布不变**。

- 💻 **PC 端部署（llama.cpp）。**
  除了服务器级的 vLLM，HunyuanOCR-1.5 还支持通过 [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 在 **CPU / 消费级 GPU / 笔记本** 上部署：使用转换后的 GGUF 权重和 OpenAI 兼容的 `llama-server`。同时我们还提供了一个适配 DFlash 的 `llama.cpp` 分支，因此同样的投机解码加速在 PC 端也可用。详见 [`docs/llama_cpp_zh.md`](docs/llama_cpp_zh.md)。

- 🧠 **更强：Agentic Data Flow + 升级的训练配方。**
  在数据侧，我们提出 **Agentic Data Flow**，一套由智能体驱动的数据构造系统，能把模型的短板转化为可执行的数据需求。智能体深度参与素材检索、基于工具的校验、样本清洗与数据流水线开发，并与算法工程师形成闭环迭代。在 HunyuanOCR-1.5 中，该系统被用于**低资源 OCR、古文字 OCR、多图文字类 QA** 等长尾能力的定向补强。
  在训练侧，我们对配方进行了系统性升级：预训练 Stage-3 被重新规划，纳入新产出的能力数据、多图数据与历史 OCR 数据，最大图像分辨率扩展到 **4K**、上下文窗口扩展到 **128K**；后训练阶段则优化 SFT 数据，并进一步在不同 OCR 任务上探索强化学习（RL），以放大 RL 带来的收益。

综合来看，HunyuanOCR-1.5 在**保留轻量化端到端模型部署优势**的同时，实现了**更快的推理与更广的 OCR 能力覆盖**。本仓库开源了 SFT / DFlash 训练流水线以及 transformers / vLLM 推理栈，方便社区复现、微调并拓展 OCR 专用 VLM。

---

## ⚙️ 环境

### 训练

- Python 3.10+（已在 3.12 上测试）
- PyTorch 2.1+（CUDA 12.1+；已完整测试 cu130 构建）
- transformers 4.57+
- DeepSpeed 0.14+

```bash
pip install -r requirements.txt
# flash-attn 需要手动编译安装：
pip install flash-attn --no-build-isolation
```

### 推理

推理现已统一为**单一环境**（基于 `uv`，需要 CUDA 13），一套环境即可运行三种配置：**vLLM AR、DFlash 投机解码、原生 transformers**，并已验证三者精度对齐。

```bash
pip install uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "vllm>=0.25.1"
uv pip install --no-build-isolation --no-cache-dir "flash-attn==2.8.3"
```

推理代码位于 [`inference/`](inference)（`inference/vLLM`、`inference/DFlash`、`inference/transformers`）。完整安装、用法，以及在没有 CUDA 13 或仅需其中一种配置时的轻量单配置方案，详见 [`docs/inference/inference_zh.md`](docs/inference/inference_zh.md)。

---

## 🚀 训练

所有训练脚本都位于 `scripts/` 目录，并共享 `scripts/env_common.sh` 中的分布式环境变量。
多机训练通过标准的 `NNODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` 环境变量支持。

### 1. 准备打包（packed）后的训练数据

我们先对每个原始 OCR JSONL 做分词，再把多个样本打包到长度上限
`packed_max_length=20480` 的单条序列中，以最大化 GPU 利用率。

**步骤 1**：在 `configs/data_list.txt` 中每行填入一个绝对路径，指向一个原始 OCR JSONL 文件。
JSONL 的数据格式说明见 [`docs/data_format_zh.md`](docs/data_format_zh.md)。

**步骤 2**：运行多进程的计数与打包流水线：

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
INPUT_LIST=./configs/data_list.txt \
PACK_LEN=20480 \
NUM_PROCESSES=32 \
THREADS_PER_PROCESS=8 \
bash scripts/pack_data.sh
```

输出 `./data/parsing_packed_20480.jsonl`：一个序列打包好、可直接用于训练的 JSONL。

该流水线的实现见 [`tools/pipeline_count_and_pack.py`](tools/pipeline_count_and_pack.py)
和 [`tools/pack_from_counted.py`](tools/pack_from_counted.py)。

### 2. 对 HunyuanOCR 基座模型进行 SFT

在打包后的 OCR 序列上进行完整的端到端 SFT（视觉编码器 + MLP + LLM）。
默认配置：`lr=2e-5`、`epochs=5`、每卡 batch=1、`packed_max_length=20480`。

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
NPROC_PER_NODE=8 \
bash scripts/sft_base.sh
```

入口：[`train/train_hunyuan.py`](train/train_hunyuan.py)。
完整参数列表见 [`docs/training_zh.md`](docs/training_zh.md)。

### 3. 从零训练 DFlash 草稿模型

训练一个小型的块扩散草稿模型，为 HunyuanOCR 预测 K 个投机 token。
默认配置：`lr=1e-4`、`epochs=2`、`num_mask_tokens=16`、`sample_block_num=8`。

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
NPROC_PER_NODE=8 \
bash scripts/sft_dflash.sh
```

入口：[`train/train_draft.py`](train/train_draft.py)。

### 4. 从已有 DFlash 检查点继续微调

当需要把已发布的 DFlash 草稿模型适配到更小 / 特定领域的数据集时使用。
推荐配置：`lr=2e-5`、`epochs=10`、`warmup_ratio=0.05`、`save_steps=500`。

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
DFLASH_INIT=/path/to/existing/dflash/checkpoint \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
NPROC_PER_NODE=8 \
bash scripts/sft_dflash_finetune.sh
```

入口：[`train/train_draft_from_dflash.py`](train/train_draft_from_dflash.py)。

---

## 🧪 推理

HunyuanOCR-1.5 在 [`inference/`](inference) 下使用**单一统一环境**，一套环境即可运行三种服务 / 推理配置：**vLLM 自回归生成、DFlash 投机解码、原生 transformers**，另加一个可选的 PC 端 llama.cpp 路径。所有配置共享同一份权重、同一套 task-type prompt + 采样 + 后处理，输出可直接横向对比，且已验证三者精度对齐。

- **vLLM AR**：自回归服务。→ [`inference/vLLM`](inference/vLLM)
- **DFlash**：AR + DFlash 投机解码，对长输出无损加速。→ [`inference/DFlash`](inference/DFlash)
- **transformers**：HuggingFace 原生自回归推理。→ [`inference/transformers`](inference/transformers)
- **llama.cpp**：CPU / 消费级 GPU / 笔记本，GGUF 部署（见下文）。

> 统一环境需要 **CUDA 13**。若没有 CUDA 13，或只需其中一种配置，可参考 [`docs/inference/inference_zh.md`](docs/inference/inference_zh.md) 里的轻量单配置方案。

### 环境安装

```bash
pip install uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "vllm>=0.25.1"
uv pip install --no-build-isolation --no-cache-dir "flash-attn==2.8.3"
```

### 下载权重

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

下载内容同时包含主模型和 `dflash/` 草稿模型。

### 快速开始（vLLM AR，单卡）

启动 OpenAI 兼容服务（对外服务名 `tencent/HunyuanOCR`、`-tp 1`、`--max-model-len 131072`）：

```bash
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/vLLM/serve.sh

# 就绪检查
curl -sf http://127.0.0.1:8000/v1/models
```

发送单张图片。提示词通过 `--task-type` 锁定为官方任务类型（`--list-tasks` 查看全部 12 种）；
采样参数（`temperature=0.0`、`top_p=1.0`、`top_k=-1`、`repetition_penalty=1.08`）
与尾部重复早停 / 清洗均已内置：

```bash
python inference/vLLM/infer_vllm_client.py \
    --image /path/to/document.png --task-type doc_parse \
    --model tencent/HunyuanOCR --port 8000 --max-tokens 32768
```

对整个目录做批量推理（多端点并发、可断点续跑）：

```bash
python inference/vLLM/batch_infer.py \
    --image-dir /path/to/images --out-dir /path/to/output \
    --ports 8000 --task-type doc_parse --max-tokens 32768 --concurrency 16
```

需要 **DFlash 加速**请用 [`inference/DFlash`](inference/DFlash)（`serve_DFlash.sh`）；
需要**原生 transformers 推理**请用 [`inference/transformers`](inference/transformers)。
完整的环境安装步骤、任务类型表和多卡说明详见 [`docs/inference/inference_zh.md`](docs/inference/inference_zh.md)。

### PC 端部署（llama.cpp）

对于 **CPU / 消费级 GPU / 笔记本** 环境，HunyuanOCR-1.5 在将权重转换为 GGUF 后，也可以通过 [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 部署。社区版 `llama.cpp`（仅支持 HunyuanOCR 基座）和一个适配 DFlash 的分支（[`wendadawen/llama.cpp @ dflash-adapt-hunyuanocr-hunyuanstyle`](https://github.com/wendadawen/llama.cpp/tree/dflash-adapt-hunyuanocr-hunyuanstyle)）都受支持。

最小化的构建与启动（社区版，不含 DFlash）：

```bash
# 1. 构建
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON     # NVIDIA GPU 追加 -DGGML_CUDA=ON
cmake --build ./build --config Release -j

# 2. 将 HunyuanOCR 转换为 GGUF（base + mmproj）
hf download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
python3 convert_hf_to_gguf.py --outfile ./HunyuanOCR/hyocr-f16.gguf        --outtype f16 ./HunyuanOCR
python3 convert_hf_to_gguf.py --outfile ./HunyuanOCR/mmproj-hyocr-f16.gguf --outtype f16 --mmproj ./HunyuanOCR

# 3. 启动服务（OpenAI 兼容）
build/bin/llama-server \
    --model  ./HunyuanOCR/hyocr-f16.gguf \
    --mmproj ./HunyuanOCR/mmproj-hyocr-f16.gguf \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096
```

适配 DFlash 的变体、草稿模型的权重转换，以及一个测试客户端（[`llama_cpp/chat.py`](llama_cpp/chat.py)，附带 [`llama_cpp/test_assets/`](llama_cpp/test_assets)下的 26 张示例 OCR 图片）：

完整指南参见 [`docs/llama_cpp_zh.md`](docs/llama_cpp_zh.md)。

---

## 📖 文档

- [`docs/training_zh.md`](docs/training_zh.md)：训练模式、超参数、分布式配置
- [`docs/data_format_zh.md`](docs/data_format_zh.md)：原始 OCR JSONL 格式与打包流水线
- [`docs/inference/inference_zh.md`](docs/inference/inference_zh.md)：统一推理环境（vLLM AR / DFlash / transformers）与部署调优
- [`docs/llama_cpp_zh.md`](docs/llama_cpp_zh.md)：使用 llama.cpp 的 PC 端部署（社区版 & DFlash 适配分支）
- [`docs/benchmark_zh.md`](docs/benchmark_zh.md)：端到端速度基准测试

---

## 📚 引用

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

## 🙏 致谢 <!-- omit in toc -->

我们衷心感谢 [Qwen](https://github.com/QwenLM/Qwen3.6) 与 [DFlash](https://github.com/z-lab/dflash)，感谢他们宝贵的模型与研究思路。

特别感谢 Hugging Face 社区提供的 Day-0 支持。

---

## 📜 许可证 <!-- omit in toc -->

HunyuanOCR-1.5 采用与 HunyuanOCR-1.0 相同的许可证：**Tencent Hunyuan Community License Agreement（腾讯混元社区许可协议）**。完整条款见 [`LICENSE`](LICENSE)。
