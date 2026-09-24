# 推理与部署指南

[English Version](./inference.md)

HunyuanOCR-1.5 现已提供**单一统一推理环境**。此前的版本使用三套独立、互斥的环境（vLLM AR / DFlash / transformers），现已整合为同一套环境，同时支持三种配置运行，**并已验证三者精度对齐**。

推理代码位于 [`inference/`](../../inference)：

- [`inference/vLLM`](../../inference/vLLM)：vLLM 自回归（AR）服务
- [`inference/DFlash`](../../inference/DFlash)：vLLM + DFlash 投机解码
- [`inference/transformers`](../../inference/transformers)：HuggingFace 原生 transformers 推理

---

## 环境安装

统一环境基于单个 `uv` 虚拟环境构建，**需要 CUDA 13**。

```bash
pip install uv

uv venv --python 3.12 && source .venv/bin/activate
uv pip install "vllm>=0.25.1" runai-model-streamer
uv pip install --no-build-isolation --no-cache-dir "flash-attn==2.8.3"
```

---

## 下载权重

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

下载内容同时包含**基座模型**与 **`dflash/` 草稿模型**（供 DFlash 路径使用）。

---

## 运行推理

`inference/vLLM/` 提供 OpenAI 兼容的 AR 服务脚本（`serve.sh`），以及共享的单图与批量客户端（`infer_vllm_client.py`、`batch_infer.py`）。DFlash 使用自己的服务脚本（`inference/DFlash/serve_DFlash.sh`），但**完全复用同一套客户端**，采样、任务提示词与后处理（`inference/utils/`）都一致，因此 AR / DFlash / transformers 三种输出可以直接对比。

```bash
# —— vLLM AR ——
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/vLLM/serve.sh

# —— vLLM + DFlash（草稿路径默认为 ${MODEL_PATH}/dflash） ——
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/DFlash/serve_DFlash.sh

# 就绪检查
curl -sf http://127.0.0.1:8000/v1/models
```

向任一服务发送单张图片（提示词通过 `--task-type` 锁定为官方任务类型，`--list-tasks` 查看全部 12 种）：

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

批量客户端会完成所有已调度图片，并将逐图结果记录到 `results.jsonl`。
只要有图片处理失败，退出码即为 **1**；全部成功时为 **0**（包括空目录或所有图片均已完成而跳过的情况）。

**原生 transformers**（多卡直接推理 / 对齐校验）：

```bash
python inference/transformers/infer_hf_8gpu.py \
    --model ./HunyuanOCR --attn-implementation flash_attention_2 \
    --input ./input.jsonl --output ./results/hf_out \
    --gpu-ids 0,1,2,3,4,5,6,7 --max-new-tokens 32768 --merge
```

---

## 没有 CUDA 13，或只需要其中一种配置？

如果你的机器没有 CUDA 13，或只需要三种配置中的**一种**（例如仅在 CUDA 12 上跑 vLLM AR，或仅跑原生 transformers），无需安装完整的统一环境。**归档中的单配置安装方案**提供了对应的轻量独立环境和已验证版本：

| 配置              | 旧独立环境            | CUDA         | 指南                                                       |
| ----------------- | --------------------- | ------------ | ---------------------------------------------------------- |
| 仅 vLLM AR        | vLLM 0.18.1（正式版） | 12.x         | [archive/vLLM_zh.md](./archive/vLLM_zh.md)                 |
| vLLM AR + DFlash  | vLLM nightly          | 13           | [archive/DFlash_zh.md](./archive/DFlash_zh.md)             |
| 原生 transformers | transformers 5.13.0   | 匹配宿主驱动 | [archive/transformers_zh.md](./archive/transformers_zh.md) |

完整选型指南、旧三环境的设计原因，以及性能调优说明，见 [`docs/inference/archive/README_zh.md`](./archive/README_zh.md)。

---

## 基准测试

端到端速度对比（AR vs DFlash 及跨模型对比）和最小复现脚本见 [`docs/benchmark_zh.md`](../benchmark_zh.md)。
