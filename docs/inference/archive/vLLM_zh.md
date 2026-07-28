# 环境 A · HunyuanOCR-1.5 with vLLM 0.18.1（CUDA 12，仅 AR） <!-- omit in toc -->

[English Version](./vLLM.md)

最省心的方案：vLLM **0.18.1（正式版）**，一条 `pip` 命令装完，原生支持 HunyuanOCR，**无需 nightly、无需 CUDA 13 compat lib、无需任何补丁**。代价是这套环境**只支持自回归（AR）推理**，正式版没有 DFlash 投机解码方法。需要 DFlash 加速请用 [`DFlash`](./DFlash_zh.md)；需要原生 transformers 推理请用 [`transformers`](./transformers_zh.md)。

> 已在 Python 3.10 + CUDA 12.8 + NVIDIA H20 上从干净环境验证。

---

## 目录 <!-- omit in toc -->

- [1. 环境安装](#1-环境安装)
- [2. 下载权重](#2-下载权重)
- [3. 启动服务（单卡）](#3-启动服务单卡)
- [4. 推理](#4-推理)
- [5. 任务类型](#5-任务类型)
- [6. 文件](#6-文件)

---

## 1. 环境安装

要求：Python 3.10、CUDA 12.x、NVIDIA GPU（显存 ≥ 24 GB；模型本身约 5 GB，其余用于 KV cache）。

```bash
conda create -n hunyuanocr python=3.10 -y
conda activate hunyuanocr
pip install "vllm==0.18.1" "openai>=1.30.0" "pillow>=10.0.0"
```

`vllm==0.18.1` 会自动拉取匹配的 `torch 2.10` / `transformers 4.57` / `flashinfer`。

> - vLLM 0.18.1 **原生支持** `HunYuanVLForConditionalGeneration`，无需 nightly、无需补丁。
> - 走公司代理时先设置：
>   `export http_proxy=http://<proxy>:<port> https_proxy=http://<proxy>:<port>`

验证安装：

```bash
python -c "from vllm.model_executor.models.registry import ModelRegistry; \
print('HunYuanVL supported:', 'HunYuanVLForConditionalGeneration' in ModelRegistry.get_supported_archs())"
# 预期：HunYuanVL supported: True
```

> 这套环境用的是 transformers 4.57.6，**不含** `hunyuan_vl` 模块（该模块需要 transformers ≥ 5.13.0）。所以它**不能**跑原生 HF transformers 推理，也不能升级 transformers（vLLM 0.18.1 要求 `transformers < 5`）。

---

## 2. 下载权重

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

这套环境只使用基座模型，不使用 `dflash/` 草稿模型。

---

## 3. 启动服务（单卡）

```bash
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash serve.sh
```

就绪检查（首次加载约 1 分钟）：

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

`serve.sh` 环境变量：`MODEL_PATH`（必填）、`GPU`（默认 0）、`PORT`（默认 8000）、`GPU_MEM_UTIL`（默认 0.9）、`MAX_MODEL_LEN`（默认 131072）、`SERVED_NAME`（默认 `tencent/HunyuanOCR`）。

> **多卡吞吐：** 每张卡启动一个实例（`GPU=0 PORT=8000`、`GPU=1 PORT=8001`、……），推理时把所有端口传给 `batch_infer.py --ports`，接近线性扩展。

停止服务：

```bash
pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "vllm serve"
```

---

## 4. 推理

### 采样参数（对齐官方设置，已内置，不需要修改） <!-- omit in toc -->

`temperature=0.0`、`top_p=1.0`、`top_k=-1`、`repetition_penalty=1.08`，流式生成 + 尾部重复早停 + 尾部重复清洗（防止贪婪解码陷入重复退化）。

### 单张图 <!-- omit in toc -->

```bash
python infer_vllm_client.py --image /path/to/doc.png --task-type doc_parse \
    --model tencent/HunyuanOCR --port 8000 --max-tokens 32768
```

- `--task-type` 选择任务（见 §5），默认 `doc_parse`。
- `doc_parse` 会自动做 markdown 规整；`--no-doc-postprocess` 关闭它。

### 批量（目录） <!-- omit in toc -->

```bash
python batch_infer.py --image-dir /path/imgs --out-dir /path/out \
    --ports 8000 --task-type doc_parse --max-tokens 32768 --concurrency 16
```

- 每张图片生成同名 `.md`；`out-dir/results.jsonl` 记录每页的延迟 / 字符数 / 早停 / 后处理明细。
- 已完成条目自动跳过（可断点续跑）。传多个端口（`--ports 8000,8001,...`）会自动轮询。

---

## 5. 任务类型

`--task-type` 选择官方推荐 prompt。列出全部：`python infer_vllm_client.py --list-tasks`

| task_type          | 说明                                                                       |
| ------------------ | -------------------------------------------------------------------------- |
| `doc_parse`        | 端到端文档解析（默认；正文 → md，表格 → HTML，公式 → LaTeX，忽略页眉页脚） |
| `structured_parse` | 结构化解析（非文档场景，如古文 / 街景）                                    |
| `spotting_json`    | 检测 + 识别 → JSON 数组（box 归一化 0-1000 + 文字）                        |
| `spotting_hunyuan` | 检测 + 识别 → Hunyuan 坐标格式                                             |
| `layout`           | 版面分析（按阅读顺序）                                                     |
| `layout_parse`     | 版面分析 + 全文解析                                                        |
| `chart_parse`      | 图表解析（流程图 → Mermaid，其他 → Markdown）                              |
| `formula`          | 公式解析（→ LaTeX）                                                        |
| `table`            | 表格解析（→ HTML）                                                         |
| `doc_trans_en2zh`  | 文档翻译，英 → 中                                                          |
| `trans_other2en`   | 通用场景翻译 → 英                                                          |
| `trans_other2zh`   | 通用场景翻译 → 中                                                          |

> Markdown 规整（由 `--no-doc-postprocess` 控制）**仅对 `doc_parse` 生效**。

---

## 6. 文件

```
inference/vLLM/
├── serve.sh                # 单卡 vLLM 启动脚本（AR）
├── infer_vllm_client.py    # 单图客户端（task_type + 后处理）
└── batch_infer.py          # 批量推理（多端点并发）
```

> 共享工具函数（`tasks.py`：task_type → 官方 prompt 映射；`output_utils.py`：流式早停 / 清洗 + doc_parse markdown 规整）单份放在 `inference/utils/`，被所有推理入口（`inference/vLLM/`、`inference/DFlash/`、`inference/transformers/`）导入。

> 图片分辨率使用模型默认（`max_pixels ≈ 4096×4096`），无需配置。`config.json` 里的 `vision_config.max_image_size` 是位置编码表形状（模型结构参数），**不要**把它当作分辨率旋钮。
