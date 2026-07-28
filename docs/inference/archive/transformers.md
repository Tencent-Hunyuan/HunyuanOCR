# Setup C · HunyuanOCR-1.5 native transformers inference (multi-GPU, no vLLM)

[中文阅读](./transformers_zh.md)

Load `HunYuanVLForConditionalGeneration` directly with HuggingFace
**transformers 5.13.0** and run inference without vLLM. Multi-GPU parallelism
(one model replica per GPU via `multiprocessing.spawn`), with sampling
**strictly aligned** to the vLLM client (greedy + `repetition_penalty=1.08` +
tail-repetition early-stop + doc_parse markdown normalization).

Use this when you need native transformers inference / alignment checks /
accuracy comparison. **For throughput and production, use a vLLM setup**
([`DFlash`](./DFlash.md) or [`vLLM`](./vLLM.md)) — transformers
has no continuous batching and processes images serially (~40–200 s per image,
markedly slower than vLLM).

> Validated on Python 3.12 + transformers 5.13.0 + torch cu128 (matching host
> driver 535, no compat) + NVIDIA H20.

---

## Contents

- [1. Why a separate environment](#1-why-a-separate-environment)
- [2. Environment setup](#2-environment-setup)
- [3. Download the weights](#3-download-the-weights)
- [4. Run inference (multi-GPU)](#4-run-inference-multi-gpu)
- [5. Sampling alignment](#5-sampling-alignment)
- [6. Files](#6-files)

---

## 1. Why a separate environment

Native transformers inference needs the `hunyuan_vl` module
(`HunYuanVLForConditionalGeneration` + `HunYuanVLProcessor`), **introduced in
transformers 5.13.0**. However:

- vLLM 0.18.1 (Setup A) requires `transformers < 5`, and 4.57.6 has no
  `hunyuan_vl` → cannot do transformers inference.
- vLLM nightly (Setup B) defaults to transformers 5.5.3, which also lacks
  `hunyuan_vl`.
- **transformers 5.13.0 breaks vLLM's HunYuanVL loading**
  (`AutoImageProcessor.register` changed its signature in 5.13 → `AttributeError`).

So **transformers inference must use a dedicated environment with transformers
5.13.0, and that environment cannot also run vLLM**. For a three-way comparison,
run vLLM AR + DFlash from [`DFlash`](./DFlash.md) and transformers here — same kernel,
weights, and sampling, so the comparison holds.

---

## 2. Environment setup

Requirements: Python 3.12, an NVIDIA GPU (≥ 24 GB VRAM; long-document generation
may peak higher).

> ✅ **CUDA version is flexible:** unlike DFlash, transformers inference is not
> tied to nightly or cu130. It just loads HunYuanVL and runs a normal forward
> pass, so the only hard requirement is **transformers ≥ 5.13.0** (the
> `hunyuan_vl` module). Install the torch that **matches your host driver** —
> usually **without** the CUDA 13 compat library, which is simpler than the
> nightly path.

**Step 1: check the CUDA version your host driver supports** (top-right of
`nvidia-smi`, `CUDA Version: X.Y`):

| Host driver                        | torch backend to install | compat needed?                            |
| ---------------------------------- | ------------------------ | ----------------------------------------- |
| CUDA 12.x (e.g. driver 535 → 12.8) | `cu128`                  | ❌ no ← validated path                    |
| CUDA 13.x / driver ≥ 580           | `cu130`                  | ❌ no                                     |
| forced to cu130 but driver < 580   | `cu130`                  | ✅ yes (see [`DFlash`](./DFlash.md) §1.2) |

**Step 2: install from scratch** (example: host `driver 535 / CUDA 12.8` →
`cu128`, no compat):

```bash
# Behind a corporate proxy: export http_proxy=http://<proxy>:<port> https_proxy=http://<proxy>:<port>
pip install -U uv
uv venv --python 3.12 && source .venv/bin/activate

# NOTE: install BOTH torch and torchvision. torchvision is a REQUIRED dependency
#   of HunYuanVLImageProcessor; omitting it raises
#   "Could not load any image processor ... Missing optional dependencies: torchvision".
uv pip install torch torchvision --torch-backend=cu128
uv pip install "transformers==5.13.0" accelerate pillow tqdm
```

> When the host driver is CUDA 13.x, swap `--torch-backend=cu128` for `cu130` —
> still no compat needed.

> - Validated versions (cu128, host driver 535, no compat): `torch 2.11.0+cu128`,
>   `torchvision 0.26.0+cu128`, `transformers 5.13.0`, `accelerate 1.14.0`,
>   `python 3.12.3`.
> - **Only if you are forced to cu130 with a host driver < 580** do you need the
>   CUDA 13 compat library — see
>   [`DFlash`](./DFlash.md) §1.2; then
>   `export LD_LIBRARY_PATH=/ABS/PATH/cuda_compat_13/extracted:$LD_LIBRARY_PATH`
>   before running.

Verify (using the native host CUDA, no `LD_LIBRARY_PATH` needed):

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())"
# expected: cuda: True | gpus: 8   (a "driver too old" error means the torch backend does not match the host driver)
python -c "from transformers import HunYuanVLForConditionalGeneration; \
from transformers.models.hunyuan_vl import processing_hunyuan_vl; print('hunyuan_vl OK')"
# expected: hunyuan_vl OK
```

---

## 3. Download the weights

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir "your/path/to/HunyuanOCR" --exclude "v1.0/*"
```

This setup uses the base model only (not the DFlash draft model).

---

## 4. Run inference (multi-GPU)

The input is a **JSONL** where each line contains an image-path field (default
key `image_path`) and an optional prompt field (default key `问题`; falls back to
`--prompt` when absent). Output is written to `--answer-key` (default
`hf_answer`), one shard JSONL per GPU, merged with `--merge`.

```bash
# Only needed if you are forced to cu130 with a host driver < 580; not needed when cu128 matches the host:
# export LD_LIBRARY_PATH=/ABS/PATH/cuda_compat_13/extracted:$LD_LIBRARY_PATH

python infer_hf_8gpu.py \
    --model  "your/path/to/HunyuanOCR" \
    --input  "your/path/to/input.jsonl" \
    --output "./results/hf_out" \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --max-new-tokens 32768 \
    --merge

# If you want to use flash-attention to accelerate the attention kernel
# uv pip install flash-attn==2.8.1 --no-cache-dir --no-build-isolation -v
# and then add the following argument to the above command:
#  --attn-implementation flash_attention_2
```

Produces `./results/hf_out_1.jsonl` … `_8.jsonl`, merged into `./results/hf_out.merged.jsonl`.

**Common arguments:**

| Argument                                        | Default                       | Description                                                                                                                      |
| ----------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `--gpu-ids`                                     | —                             | Physical GPU ids, e.g. `0,1,2,3`; overrides `--num-gpus`                                                                         |
| `--num-gpus`                                    | 8                             | First N GPUs when `--gpu-ids` is not given                                                                                       |
| `--max-new-tokens`                              | 32768                         | **Recommend 8192**: on repetition degeneration greedy decoding runs to the cap, so 8192 fails faster (early-stop also truncates) |
| `--repetition-penalty`                          | 1.08                          | Aligned with vLLM                                                                                                                |
| `--repeat-min-repeats`                          | 8                             | Tail-repetition early-stop threshold                                                                                             |
| `--no-stream`                                   | off                           | Disables tail-repetition early-stop (single-shot generation)                                                                     |
| `--no-doc-postprocess`                          | off                           | Disables doc_parse markdown normalization (on by default, aligned with vLLM)                                                     |
| `--image-key` / `--prompt-key` / `--answer-key` | image_path / 问题 / hf_answer | Input/output field names                                                                                                         |
| `--merge`                                       | —                             | Merge per-GPU shards after the run                                                                                               |
| `--no-resume`                                   | —                             | Disable resume (by default completed lines are skipped)                                                                          |

> **max-new-tokens note:** transformers `generate()` has no mid-stream stop like
> vLLM, so once it enters repetition degeneration it runs to `max_new_tokens`
> (tens of minutes for a single image). The script has a built-in
> `TailRepetitionStop` early-stop + `clean_repeated_substrings` fallback, but
> setting the cap to **8192** (rather than 32768) is still recommended.

---

## 5. Sampling alignment

The generation settings are **strictly aligned** with the vLLM client (see the
script's header docstring):

|                            | Value              | Alignment                                                                                       |
| -------------------------- | ------------------ | ----------------------------------------------------------------------------------------------- |
| temperature                | 0.0                | `do_sample=False` (greedy)                                                                      |
| repetition_penalty         | 1.08               | `generate(repetition_penalty=1.08)`                                                             |
| tail-repetition early-stop | ✅                 | `TailRepetitionStop` StoppingCriteria, replicating vLLM's streaming `has_tail_repetition` logic |
| tail-repetition cleanup    | ✅                 | `clean_repeated_substrings` (final fallback)                                                    |
| doc_parse normalization    | ✅ (on by default) | `output_utils.normalize_doc_parse_markdown`, shared with the vLLM client                        |
| skip_special_tokens        | True               | `batch_decode(skip_special_tokens=True)`                                                        |

Model loading follows the official HunyuanOCR HuggingFace inference recipe:
`HunYuanVLForConditionalGeneration` + `AutoProcessor`, `dtype=bfloat16`,
`attn_implementation=eager`, including the video-token patch for older tokenizer
snapshots.

---

## 6. Files

```
inference/transformers/
└── infer_hf_8gpu.py   # multi-GPU transformers inference (self-contained: load/early-stop/cleanup)
```

> The doc_parse markdown normalization (`normalize_doc_parse_markdown`) is imported from
> `inference/utils/output_utils.py` — a single shared copy across
> `inference/vLLM/`, `inference/DFlash/`, and `inference/transformers/`.

> Image resolution uses the model default (`max_pixels ≈ 4096×4096`); no
> configuration needed.
