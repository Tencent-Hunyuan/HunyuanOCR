# 环境 C · HunyuanOCR-1.5 原生 transformers 推理（多卡，不经 vLLM）<!-- omit in toc -->

[English Version](./transformers.md)

用 HuggingFace **transformers 5.13.0** 直接加载 `HunYuanVLForConditionalGeneration` 做推理，不经过 vLLM。多卡并行由 `multiprocessing.spawn` 每卡起一份模型副本实现，采样参数**严格对齐** vLLM 客户端（greedy + `repetition_penalty=1.08` + 尾部重复早停 + doc_parse markdown 规整）。

适合场景：需要原生 transformers 推理 / 对齐校验 / 精度对比。**追求吞吐或线上部署，请用 vLLM 方案**（[`DFlash`](./DFlash_zh.md) 或 [`vLLM`](./vLLM_zh.md)），transformers 没有 continuous batching，图片是串行处理的（每张约 40–200 秒，明显比 vLLM 慢）。

> 已在 Python 3.12 + transformers 5.13.0 + torch cu128（匹配宿主驱动 535，无需 compat）+ NVIDIA H20 上验证。

---

## 目录 <!-- omit in toc -->

- [1. 为什么要单独一套环境](#1-为什么要单独一套环境)
- [2. 环境安装](#2-环境安装)
- [3. 下载权重](#3-下载权重)
- [4. 运行推理（多卡）](#4-运行推理多卡)
- [5. 采样对齐](#5-采样对齐)
- [6. 文件](#6-文件)

---

## 1. 为什么要单独一套环境

原生 transformers 推理需要 `hunyuan_vl` 模块（`HunYuanVLForConditionalGeneration` + `HunYuanVLProcessor`），**该模块在 transformers 5.13.0 中引入**。但是：

- vLLM 0.18.1（环境 A）要求 `transformers < 5`，4.57.6 没有 `hunyuan_vl`，无法做 transformers 推理。
- vLLM nightly（环境 B）默认用 transformers 5.5.3，同样没有 `hunyuan_vl`。
- **transformers 5.13.0 会破坏 vLLM 的 HunYuanVL 加载**（`AutoImageProcessor.register` 在 5.13 改了签名，导致 `AttributeError`）。

所以 **transformers 推理必须用一个 transformers 5.13.0 的专用环境，且该环境不能同时跑 vLLM**。若要做三方对比，从 [`DFlash`](./DFlash_zh.md) 跑 vLLM AR + DFlash，从这里跑 transformers。三者内核、权重、采样都一致，对比方法学上成立。

---

## 2. 环境安装

要求：Python 3.12、NVIDIA GPU（显存 ≥ 24 GB；长文档生成峰值可能更高）。

> ✅ **CUDA 版本比较灵活：** 与 DFlash 不同，transformers 推理并不绑定 nightly 或 cu130。它只加载 HunYuanVL 做常规前向，所以硬性要求只有 **transformers ≥ 5.13.0**（`hunyuan_vl` 模块）。装一个**匹配宿主驱动**的 torch 即可，通常**不需要**CUDA 13 compat 库，比 nightly 那套简单得多。

**Step 1：查宿主驱动支持的 CUDA 版本**（`nvidia-smi` 右上角的 `CUDA Version: X.Y`）：

| 宿主驱动                       | 应装的 torch backend | 是否需要 compat？                             |
| ------------------------------ | -------------------- | --------------------------------------------- |
| CUDA 12.x（如驱动 535 → 12.8） | `cu128`              | ❌ 不需要 ← 已验证路径                        |
| CUDA 13.x / 驱动 ≥ 580         | `cu130`              | ❌ 不需要                                     |
| 被迫用 cu130 但驱动 < 580      | `cu130`              | ✅ 需要（见 [`DFlash`](./DFlash_zh.md) §1.2） |

**Step 2：从零安装**（示例：宿主 `驱动 535 / CUDA 12.8` → `cu128`，不需要 compat）：

```bash
# 走公司代理：export http_proxy=http://<proxy>:<port> https_proxy=http://<proxy>:<port>
pip install -U uv
uv venv --python 3.12 && source .venv/bin/activate

# 注意：torch 和 torchvision 都要装。torchvision 是 HunYuanVLImageProcessor 的必需依赖，
#   缺了会报 "Could not load any image processor ... Missing optional dependencies: torchvision"。
uv pip install torch torchvision --torch-backend=cu128
uv pip install "transformers==5.13.0" accelerate pillow tqdm
```

> 若宿主驱动是 CUDA 13.x，把 `--torch-backend=cu128` 换成 `cu130` 即可，仍然不需要 compat。

> - 已验证版本（cu128、宿主驱动 535、无 compat）：`torch 2.11.0+cu128`、`torchvision 0.26.0+cu128`、`transformers 5.13.0`、`accelerate 1.14.0`、`python 3.12.3`。
> - **只有当你被迫用 cu130 且宿主驱动 < 580 时**，才需要 CUDA 13 compat 库，详见 [`DFlash`](./DFlash_zh.md) §1.2，然后在运行前 `export LD_LIBRARY_PATH=/ABS/PATH/cuda_compat_13/extracted:$LD_LIBRARY_PATH`。

验证（使用宿主原生 CUDA，无需设置 `LD_LIBRARY_PATH`）：

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())"
# 预期：cuda: True | gpus: 8  （若报 "driver too old" 说明 torch backend 与宿主驱动不匹配）
python -c "from transformers import HunYuanVLForConditionalGeneration; \
from transformers.models.hunyuan_vl import processing_hunyuan_vl; print('hunyuan_vl OK')"
# 预期：hunyuan_vl OK
```

---

## 3. 下载权重

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir "your/path/to/HunyuanOCR" --exclude "v1.0/*"
```

本环境只使用基座模型，不使用 DFlash 草稿模型。

---

## 4. 运行推理（多卡）

输入是一个 **JSONL**，每行含一个图片路径字段（默认 key `image_path`）和一个可选的 prompt 字段（默认 key `问题`；缺失时回退到 `--prompt`）。输出写到 `--answer-key`（默认 `hf_answer`），每张 GPU 一个分片 JSONL，可用 `--merge` 合并。

```bash
# 仅当你被迫用 cu130 且宿主驱动 < 580 时需要；cu128 匹配宿主时不需要：
# export LD_LIBRARY_PATH=/ABS/PATH/cuda_compat_13/extracted:$LD_LIBRARY_PATH

python infer_hf_8gpu.py \
    --model  "your/path/to/HunyuanOCR" \
    --input  "your/path/to/input.jsonl" \
    --output "./results/hf_out" \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --max-new-tokens 32768 \
    --merge

# 若希望用 flash-attention 加速 attention kernel：
# uv pip install flash-attn==2.8.1 --no-cache-dir --no-build-isolation -v
# 然后在上面命令中追加：
#  --attn-implementation flash_attention_2
```

生成 `./results/hf_out_1.jsonl` … `_8.jsonl`，合并为 `./results/hf_out.merged.jsonl`。

**常用参数：**

| 参数                                            | 默认值                        | 说明                                                                                   |
| ----------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------- |
| `--gpu-ids`                                     | —                             | 物理 GPU id，例如 `0,1,2,3`；会覆盖 `--num-gpus`                                       |
| `--num-gpus`                                    | 8                             | 未指定 `--gpu-ids` 时使用前 N 张卡                                                     |
| `--max-new-tokens`                              | 32768                         | **推荐 8192**：进入重复退化后贪婪解码会一直生成到上限，8192 能更快失败（早停也会截断） |
| `--repetition-penalty`                          | 1.08                          | 与 vLLM 对齐                                                                           |
| `--repeat-min-repeats`                          | 8                             | 尾部重复早停阈值                                                                       |
| `--no-stream`                                   | off                           | 关闭尾部重复早停（改为一次性生成）                                                     |
| `--no-doc-postprocess`                          | off                           | 关闭 doc_parse markdown 规整（默认打开，与 vLLM 对齐）                                 |
| `--image-key` / `--prompt-key` / `--answer-key` | image_path / 问题 / hf_answer | 输入 / 输出字段名                                                                      |
| `--merge`                                       | —                             | 运行后合并各卡分片                                                                     |
| `--no-resume`                                   | —                             | 关闭断点续跑（默认会跳过已完成行）                                                     |

> **关于 max-new-tokens：** transformers 的 `generate()` 没有 vLLM 那种中途停止机制，一旦进入重复退化就会跑到 `max_new_tokens`（单图可能几十分钟）。脚本内置了 `TailRepetitionStop` 早停和 `clean_repeated_substrings` 兜底，即便如此，把上限设为 **8192**（而不是 32768）仍然是推荐做法。

---

## 5. 采样对齐

生成配置与 vLLM 客户端**严格对齐**（脚本头 docstring 也写了）：

|                     | 值             | 对齐方式                                                                           |
| ------------------- | -------------- | ---------------------------------------------------------------------------------- |
| temperature         | 0.0            | `do_sample=False`（贪婪）                                                          |
| repetition_penalty  | 1.08           | `generate(repetition_penalty=1.08)`                                                |
| 尾部重复早停        | ✅             | `TailRepetitionStop` StoppingCriteria，复刻 vLLM 流式的 `has_tail_repetition` 逻辑 |
| 尾部重复清洗        | ✅             | `clean_repeated_substrings`（最终兜底）                                            |
| doc_parse 规整      | ✅（默认打开） | `output_utils.normalize_doc_parse_markdown`，与 vLLM 客户端共用                    |
| skip_special_tokens | True           | `batch_decode(skip_special_tokens=True)`                                           |

模型加载沿用 HunyuanOCR 官方 HuggingFace 推理示例的做法：`HunYuanVLForConditionalGeneration` + `AutoProcessor`，`dtype=bfloat16`，`attn_implementation=eager`，并对老版 tokenizer 快照做了 video-token 补丁。

---

## 6. 文件

```
inference/transformers/
└── infer_hf_8gpu.py   # 多卡 transformers 推理（自包含：加载 / 早停 / 清洗）
```

> doc_parse markdown 规整（`normalize_doc_parse_markdown`）从 `inference/utils/output_utils.py` 导入，三个环境（`inference/vLLM/`、`inference/DFlash/`、`inference/transformers/`）共用同一份。

> 图片分辨率使用模型默认（`max_pixels ≈ 4096×4096`），无需配置。
