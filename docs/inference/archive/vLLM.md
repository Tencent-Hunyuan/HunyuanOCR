# Setup A · HunyuanOCR-1.5 with vLLM 0.18.1 (CUDA 12, AR only)

[中文阅读](./vLLM_zh.md)

The simplest setup: vLLM **0.18.1 (release)**, installed with a single `pip`
command, with native HunyuanOCR support — **no nightly, no CUDA 13 compat
library, no patching**. The trade-off is that it supports **autoregressive (AR)
inference only** — the release build does not include the DFlash
speculative-decoding method. For DFlash acceleration use [`DFlash`](./DFlash.md);
for native transformers inference use [`transformers`](./transformers.md).

> Validated from a clean install on Python 3.10 + CUDA 12.8 + NVIDIA H20.

---

## Contents

- [1. Environment setup](#1-environment-setup)
- [2. Download the weights](#2-download-the-weights)
- [3. Start the server (single GPU)](#3-start-the-server-single-gpu)
- [4. Inference](#4-inference)
- [5. Task types](#5-task-types)
- [6. Files](#6-files)

---

## 1. Environment setup

Requirements: Python 3.10, CUDA 12.x, an NVIDIA GPU (≥ 24 GB VRAM; the model is
~5 GB, the rest is for the KV cache).

```bash
conda create -n hunyuanocr python=3.10 -y
conda activate hunyuanocr
pip install "vllm==0.18.1" "openai>=1.30.0" "pillow>=10.0.0"
```

`vllm==0.18.1` pulls the matching `torch 2.10` / `transformers 4.57` /
`flashinfer` automatically.

> - vLLM 0.18.1 **natively supports** `HunYuanVLForConditionalGeneration` — no
>   nightly, no patch.
> - Behind a corporate proxy:
>   `export http_proxy=http://<proxy>:<port> https_proxy=http://<proxy>:<port>`

Verify the install:

```bash
python -c "from vllm.model_executor.models.registry import ModelRegistry; \
print('HunYuanVL supported:', 'HunYuanVLForConditionalGeneration' in ModelRegistry.get_supported_archs())"
# expected: HunYuanVL supported: True
```

> This setup uses transformers 4.57.6, which does **not** include the
> `hunyuan_vl` module (that needs transformers ≥ 5.13.0). It therefore **cannot**
> run native HuggingFace transformers inference, and transformers cannot be
> upgraded (vLLM 0.18.1 requires `transformers < 5`).

---

## 2. Download the weights

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

This setup uses the base model only (not the `dflash/` draft model).

---

## 3. Start the server (single GPU)

```bash
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash serve.sh
```

Readiness check (first load takes ~1 min):

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

`serve.sh` environment variables: `MODEL_PATH` (required) / `GPU` (default 0) /
`PORT` (default 8000) / `GPU_MEM_UTIL` (default 0.9) / `MAX_MODEL_LEN`
(default 131072) / `SERVED_NAME` (default `tencent/HunyuanOCR`).

> **Multi-GPU throughput:** launch one instance per GPU (`GPU=0 PORT=8000`,
> `GPU=1 PORT=8001`, …) and pass all ports to `batch_infer.py --ports` at
> inference time for near-linear scaling.

Stop the server:

```bash
pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "vllm serve"
```

---

## 4. Inference

### Sampling parameters (aligned with the official settings, built in, do not change)

`temperature=0.0`, `top_p=1.0`, `top_k=-1`, `repetition_penalty=1.08`, streaming
generation + tail-repetition early-stop + tail-repetition cleanup (to prevent
greedy-decoding repetition degeneration).

### Single image

```bash
python infer_vllm_client.py --image /path/to/doc.png --task-type doc_parse \
    --model tencent/HunyuanOCR --port 8000 --max-tokens 32768
```

- `--task-type` selects the task (see §5); default `doc_parse`.
- `doc_parse` applies markdown normalization automatically; `--no-doc-postprocess`
  disables it.

### Batch (directory)

```bash
python batch_infer.py --image-dir /path/imgs --out-dir /path/out \
    --ports 8000 --task-type doc_parse --max-tokens 32768 --concurrency 16
```

- Each image produces a same-named `.md`; `out-dir/results.jsonl` records
  per-page latency / char count / early-stop / post-processing details.
- Completed items are skipped automatically (resumable). Multiple instances
  (`--ports 8000,8001,...`) are round-robined automatically.

---

## 5. Task types

`--task-type` selects the official recommended prompt. List them all:
`python infer_vllm_client.py --list-tasks`

| task_type          | Description                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| `doc_parse`        | End-to-end document parsing (default; body → md, tables → HTML, formulas → LaTeX, headers/footers ignored) |
| `structured_parse` | Structured parsing (non-document scenes such as ancient text / street view)                                |
| `spotting_json`    | Detection + recognition → JSON array (box normalized 0-1000 + text)                                        |
| `spotting_hunyuan` | Detection + recognition → Hunyuan coordinate format                                                        |
| `layout`           | Layout analysis (in reading order)                                                                         |
| `layout_parse`     | Layout analysis + full-text parsing                                                                        |
| `chart_parse`      | Chart parsing (flowcharts → Mermaid, others → Markdown)                                                    |
| `formula`          | Formula parsing (→ LaTeX)                                                                                  |
| `table`            | Table parsing (→ HTML)                                                                                     |
| `doc_trans_en2zh`  | Document translation, English → Chinese                                                                    |
| `trans_other2en`   | General-scene translation → English                                                                        |
| `trans_other2zh`   | General-scene translation → Chinese                                                                        |

> Markdown normalization (controlled by `--no-doc-postprocess`) applies **only to
> `doc_parse`**.

---

## 6. Files

```
inference/vLLM/
├── serve.sh                # single-GPU vLLM launch script (AR)
├── infer_vllm_client.py    # single-image client (task_type + post-processing)
└── batch_infer.py          # batch inference (multi-endpoint concurrency)
```

> Shared helpers (`tasks.py` = task_type → official prompt mapping,
> `output_utils.py` = streaming early-stop / cleanup + doc_parse markdown
> normalization) live in a single copy at `inference/utils/` and are imported by
> every inference entry point (`inference/vLLM/`, `inference/DFlash/`,
> `inference/transformers/`).

> Image resolution uses the model default (`max_pixels ≈ 4096×4096`); no
> configuration needed. The `vision_config.max_image_size` in `config.json` is
> the positional-encoding table shape (a model-structure parameter) and **must
> not** be treated as a resolution knob.
