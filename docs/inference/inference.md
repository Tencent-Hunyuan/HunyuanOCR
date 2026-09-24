# Inference & Deployment Guide

[中文阅读](./inference_zh.md)

HunyuanOCR-1.5 now ships a **single unified inference environment**. The earlier
release used three separate, mutually exclusive setups (vLLM AR / DFlash /
transformers); those have since been consolidated into one environment that
runs all three, with **accuracy alignment verified across them**.

The inference code lives under [`inference/`](../../inference):

- [`inference/vLLM`](../../inference/vLLM) — vLLM autoregressive (AR) serving
- [`inference/DFlash`](../../inference/DFlash) — vLLM + DFlash speculative decoding
- [`inference/transformers`](../../inference/transformers) — native HuggingFace transformers inference

---

## Environment setup

The unified environment is built on a single `uv` virtual environment. It **requires CUDA 13**.

```bash
pip install uv

uv venv --python 3.12 && source .venv/bin/activate
uv pip install "vllm>=0.25.1" runai-model-streamer
uv pip install --no-build-isolation --no-cache-dir "flash-attn==2.8.3"
```

---

## Download the weights

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

The download contains both the **base model** and the **`dflash/` draft model** (used by the DFlash path).

---

## Run inference

`inference/vLLM/` provides the OpenAI-compatible AR server (`serve.sh`) plus the
shared single-image and batch clients (`infer_vllm_client.py`, `batch_infer.py`).
The DFlash setup uses its own server script (`inference/DFlash/serve_DFlash.sh`)
but **reuses the exact same clients** — sampling, task prompts, and
post-processing (`inference/utils/`) are identical, so AR / DFlash / transformers
outputs are directly comparable.

```bash
# —— vLLM AR ——
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/vLLM/serve.sh

# —— vLLM + DFlash (draft defaults to ${MODEL_PATH}/dflash) ——
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/DFlash/serve_DFlash.sh

# readiness check
curl -sf http://127.0.0.1:8000/v1/models
```

Send a single image against either server (prompt locked to an official task
type via `--task-type`; run `--list-tasks` for all 12):

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

The batch client finishes all scheduled images and records per-image results in
`results.jsonl`. It exits with status **1** if any image fails, or **0** when all
scheduled images succeed (including empty or fully resumed batches).

For **native transformers** (multi-GPU direct inference / alignment checks):

```bash
python inference/transformers/infer_hf_8gpu.py \
    --model ./HunyuanOCR --attn-implementation flash_attention_2 \
    --input ./input.jsonl --output ./results/hf_out \
    --gpu-ids 0,1,2,3,4,5,6,7 --max-new-tokens 32768 --merge
```

---

## No CUDA 13, or only need one configuration?

If your machine does not have CUDA 13, or you only need **one** of the three
configurations (e.g. just vLLM AR on CUDA 12, or just native transformers), you
do not have to install the full unified environment. The **archived
per-configuration recipes** cover the lighter, standalone environments and the
exact validated versions for each:

| Configuration       | Old standalone env    | CUDA        | Guide                                                |
| ------------------- | --------------------- | ----------- | ---------------------------------------------------- |
| vLLM AR only        | vLLM 0.18.1 (release) | 12.x        | [archive/vLLM.md](./archive/vLLM.md)                 |
| vLLM AR + DFlash    | vLLM nightly          | 13          | [archive/DFlash.md](./archive/DFlash.md)             |
| Native transformers | transformers 5.13.0   | host driver | [archive/transformers.md](./archive/transformers.md) |

See [`docs/inference/archive/README.md`](./archive/README.md) for the
full selection guide, the rationale behind the old three-environment split, and
performance-tuning notes.

---

## Benchmarking

See [`docs/benchmark.md`](../benchmark.md) for the full speed comparison
(AR vs DFlash and cross-model) and a minimal reproduction script.
