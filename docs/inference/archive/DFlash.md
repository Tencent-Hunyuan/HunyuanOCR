# Setup B · HunyuanOCR-1.5 with vLLM nightly (CUDA 13, AR + DFlash)

[中文阅读](./DFlash_zh.md)

The vLLM **nightly** setup supports both **autoregressive (AR)** decoding and
**DFlash speculative decoding** (lossless acceleration; the longer the output,
the larger the gain). The DFlash draft model lives in the `dflash/` subfolder of
the HF model repo and is downloaded together with the base model (see §2). The
environment is heavier than Setup A: it needs CUDA 13
(torch cu130) plus a CUDA 13 compat library. For native transformers inference
use [`transformers`](./transformers.md) (the transformers 5.5.3 in this setup
does not support it).

> Validated from a clean install on Python 3.12 + vLLM nightly (cu130) +
> NVIDIA H20 (driver 535 + a self-built CUDA 13 compat library), including a full
> OmniDocBench run (1651 images each for AR and DFlash). The three ⚠️ steps in §1
> are exactly the pitfalls found during that walkthrough — do not skip them.

**AR or DFlash?**

- Outputs match (DFlash is lossless acceleration); sampling is identical and the
  clients are shared verbatim.
- DFlash speedup grows with output length (significant for long document
  parsing, roughly on par for short outputs) and is larger at lower concurrency.
- Want the fastest long-document throughput → DFlash; want the simplest baseline
  → AR. Just switch between the two serve scripts.

---

## Contents

- [1. Environment setup (important — more involved than Setup A, follow step by step)](#1-environment-setup)
- [2. Download the weights + draft model](#2-download-the-weights--draft-model)
- [3. Start the server (single GPU)](#3-start-the-server-single-gpu)
- [4. Inference](#4-inference)
- [5. Task types](#5-task-types)
- [6. Files](#6-files)

---

## 1. Environment setup

Requirements: Python 3.12, an NVIDIA GPU (≥ 24 GB VRAM). **The DFlash
speculative method is registered only in vLLM nightly (cu130)** — release builds
0.18.1 / 0.24.0 do not support it, so the nightly build is mandatory.

> ⚠️ **This section has three mandatory steps (the two in §1.1 plus the compat
> library in §1.2). Missing any one of them prevents the server from starting.**
> These pitfalls come from the fact that the nightly build rolls its
> dependencies daily; each step below explains _why_, so don't skip.

### 1.1 Install vLLM nightly (cu130)

Behind a corporate proxy, export it first:
`export http_proxy=http://<proxy>:<port> https_proxy=http://<proxy>:<port>`

```bash
pip install -U uv
uv venv --python 3.12 && source .venv/bin/activate

# Step 1: install vLLM nightly (cu130); pulls torch cu130 / transformers / flashinfer / torchcodec
uv pip install -U vllm --torch-backend=cu130 --extra-index-url https://wheels.vllm.ai/nightly
uv pip install openai pillow
```

#### ⚠️ Step 2 (required): pin transformers to 5.5.3

This is the biggest pitfall. Installing nightly **pulls transformers 5.13.0 by
default**, but **5.13.0 crashes vLLM when loading HunYuanVL**:

```
AttributeError: 'str' object has no attribute '__module__'
  (vLLM's hunyuan_vl_image.py calls AutoImageProcessor.register("...", ...) with a
   string; transformers 5.13 changed the register() signature and calls .__module__
   on the first argument.)
```

`vllm serve` fails to start, and the traceback does not necessarily point at
transformers, making it hard to diagnose. **Downgrade explicitly to 5.5.3**
(`--no-deps` keeps torch/vLLM untouched):

```bash
uv pip install "transformers==5.5.3" --no-deps
```

> Verify: `python -c "import transformers; print(transformers.__version__)"`
> should print `5.5.3`. If you need native HuggingFace transformers inference
> (which requires transformers 5.13.0), that is a **separate, independent
> environment** — see [`transformers`](./transformers.md). It cannot share this
> vLLM environment.

#### ⚠️ Step 3 (required): uninstall torchcodec

Nightly also pulls `torchcodec` (for video decoding). It loads low-level FFmpeg
libraries (`libavutil.so.56`, …) at import time, and a missing library raises
`OSError` (not `ImportError`), which vLLM's try/except does not catch — crashing
`vllm serve`:

```
OSError: libavutil.so.56: cannot open shared object file
  → Could not load .../torchcodec/libtorchcodec_core4.so
```

HunyuanOCR is an **image** model and does not need video decoding. Just remove
it, so vLLM takes its ImportError fallback path:

```bash
uv pip uninstall torchcodec
```

> Validated versions: `vllm 0.23.1rc1.dev825~dev869` (the suffix rolls daily),
> `torch 2.11.0+cu130`, **`transformers 5.5.3`**, `flashinfer 0.6.13`, torchcodec
> uninstalled. The cu129 nightly does not work; cu130 does.

### 1.2 CUDA 13 compat library (required when the host driver is < 580) ⚠️

`torch cu130` needs a **≥ 580** user-space CUDA driver library (`libcuda.so`). If
the host driver is older (e.g. `nvidia-smi` shows 535.x / CUDA 12.8), startup
fails with:

```
The NVIDIA driver on your system is too old (found version 12080)...
```

Fix: install `cuda-compat-13-0` and prepend its `libcuda.so.580.x` via
`LD_LIBRARY_PATH` (this touches neither the host driver nor requires root). If
the host driver is already ≥ 580, skip this section.

#### Step 1: download the rpm

Download from the **official NVIDIA CUDA repo** (RHEL 8 / x86_64). First list the
available versions:

```bash
# List all available cuda-compat-13-0 versions (NVIDIA prunes old ones; any 580.x satisfies ">=580")
curl -sSL "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/" \
    | grep -oE 'cuda-compat-13-0-[0-9.]+-1\.el8\.x86_64\.rpm' | sort -u
```

Pick one (example uses `580.65.06`; any version listed above works):

```bash
BASE="https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64"
RPM="cuda-compat-13-0-580.65.06-1.el8.x86_64.rpm"
mkdir -p cuda_compat_13 && curl -sSL "$BASE/$RPM" -o "cuda_compat_13/$RPM"
```

> For other systems, swap the repo path: `rhel9`, `ubuntu2204/x86_64` (.deb),
> etc. See `https://developer.download.nvidia.com/compute/cuda/repos/`.

#### Step 2: extract the rpm

**Option A (when rpm2cpio/cpio are available):**

```bash
cd cuda_compat_13 && rpm2cpio cuda-compat-13-0-*.x86_64.rpm | cpio -idmv && cd ..
# extracted libraries are under cuda_compat_13/usr/local/cuda-13.0/compat/
mkdir -p cuda_compat_13/extracted
cp -a cuda_compat_13/usr/local/cuda-13.0/compat/* cuda_compat_13/extracted/
```

**Option B (no rpm2cpio/cpio — a pure-Python extractor; newer machines often
lack those tools):**

```bash
python3 - <<'PYEOF'
import struct, os, lzma
rpm = [f for f in os.listdir("cuda_compat_13") if f.endswith(".rpm")][0]
data = open(f"cuda_compat_13/{rpm}", "rb").read()
# skip 96B lead + two headers (magic 8e ad e8), locate the payload
def hdr_end(buf, off):
    assert buf[off:off+3] == b'\x8e\xad\xe8'
    ni, ns = struct.unpack(">II", buf[off+8:off+16])
    return off + 16 + ni*16 + ns
pos = hdr_end(data, 96); pos = (pos+7) & ~7; pos = hdr_end(data, pos)
payload = data[pos:]                              # xz-compressed cpio (newc)
raw = lzma.decompress(payload) if payload[:6]==b'\xfd7zXZ\x00' else payload
os.makedirs("cuda_compat_13/extracted", exist_ok=True)
p = 0
while p < len(raw) and raw[p:p+6] == b'070701':
    g = lambda i: int(raw[p+6+i*8:p+14+i*8], 16)
    nsz, fsz, mode = g(11), g(6), g(1)
    name = raw[p+110:p+110+nsz-1].decode("utf-8","replace")
    doff = (p+110+nsz+3) & ~3
    fdata = raw[doff:doff+fsz]
    p = (doff+fsz+3) & ~3
    if name == "TRAILER!!!": break
    base = os.path.basename(name)
    if "compat" in name and ".so" in base:
        dst = f"cuda_compat_13/extracted/{base}"
        if (mode & 0xf000) == 0xa000:            # symlink
            if os.path.lexists(dst): os.remove(dst)
            os.symlink(fdata.decode(), dst)
        elif fsz > 0:
            open(dst, "wb").write(fdata)
print("extracted:", sorted(os.listdir("cuda_compat_13/extracted")))
PYEOF
```

Either way, `cuda_compat_13/extracted/` should then contain `libcuda.so.580.x`,
`libnvidia-ptxjitcompiler.so.580.x`, `libnvidia-nvvm.so.580.x`, etc., plus their
symlinks.

#### Step 3: prepend to LD_LIBRARY_PATH before every launch

```bash
export LD_LIBRARY_PATH=$(pwd)/cuda_compat_13/extracted:$LD_LIBRARY_PATH
```

### 1.3 Verify the install

```bash
export LD_LIBRARY_PATH=$(pwd)/cuda_compat_13/extracted:$LD_LIBRARY_PATH

# a) compat works: torch sees the GPU (no more "driver too old")
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())"
# expected: cuda: True | gpus: 8

# b) the dflash method is registered
python -c "from vllm.config import SpeculativeConfig; import inspect; \
print('dflash registered:', 'dflash' in inspect.getsource(SpeculativeConfig))"
# expected: dflash registered: True

# c) transformers is 5.5.3 (not 5.13.0)
python -c "import transformers; print('transformers:', transformers.__version__)"
# expected: transformers: 5.5.3
```

---

## 2. Download the weights + draft model

```bash
pip install -U "huggingface_hub[cli]"
# base model + DFlash draft in one shot: the draft lives in the dflash/ subfolder
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

The HF model repo ships the DFlash draft (`config.json` + `dflash.py` +
tokenizer + `model.safetensors`) under the `dflash/` subfolder, so the single
download above pulls both the base model and the draft. `serve_DFlash.sh` then
uses `${MODEL_PATH}/dflash` as the default `DFLASH_PATH`; override `DFLASH_PATH`
only if you keep the draft elsewhere.

---

## 3. Start the server (single GPU)

> AR and DFlash are launched with the same server scripts that ship with the
> current unified layout — `inference/vLLM/serve.sh` for AR and
> `inference/DFlash/serve_DFlash.sh` for DFlash. Their internals (vLLM flags,
> sampling defaults, DFlash `--speculative-config`) match what this old
> nightly setup used, so the recipe below still applies.

```bash
# ⚠️ Set the compat library first when the host driver is old (see §1.2)
export LD_LIBRARY_PATH=/ABS/PATH/cuda_compat_13/extracted:$LD_LIBRARY_PATH

# —— AR (autoregressive) ——
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/vLLM/serve.sh

# —— DFlash (speculative decoding) ——
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/DFlash/serve_DFlash.sh
```

Readiness: AR ~1-2 min; **DFlash's first load includes torch.compile, ~3-5 min**.

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

**`inference/vLLM/serve.sh`** variables: `MODEL_PATH` (required) / `GPU` / `PORT` /
`GPU_MEM_UTIL` (default 0.9) / `MAX_MODEL_LEN` / `SERVED_NAME`.
**`inference/DFlash/serve_DFlash.sh`** takes the same set plus `DFLASH_PATH`
(default `${MODEL_PATH}/dflash`) / `NUM_SPEC_TOKENS` (default 15); it sets
`GPU_MEM_UTIL` to 0.85 by default to leave headroom for the draft (~0.7 GB).
Under the hood, DFlash adds one flag over AR:
`--speculative-config '{"method":"dflash","model":"<DFLASH_PATH>","num_speculative_tokens":15}'`.

> **Multi-GPU (full 8-GPU):** launch one instance per GPU (`GPU=0 PORT=8000` …
> `GPU=7 PORT=8007`, each with `LD_LIBRARY_PATH`), then run
> `python inference/vLLM/batch_infer.py --ports 8000,8001,...,8007`.

Stop the server: `pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "vllm serve"`

---

## 4. Inference

> AR and DFlash share the same clients — `inference/vLLM/infer_vllm_client.py`
> (single image) and `inference/vLLM/batch_infer.py` (batch). They import the
> shared task prompts + output utilities from
> `inference/utils/tasks.py` and `inference/utils/output_utils.py`
> (a single copy). Sampling parameters, task prompts, and post-processing are
> therefore identical, so AR / DFlash / transformers outputs are directly
> comparable.

### Sampling parameters (aligned with the official settings, built in, do not change)

`temperature=0.0`, `top_p=1.0`, `top_k=-1`, `repetition_penalty=1.08`, streaming
generation + tail-repetition early-stop + tail-repetition cleanup.

### Single image

```bash
python inference/vLLM/infer_vllm_client.py --image /path/doc.png --task-type doc_parse \
    --model tencent/HunyuanOCR --port 8000 --max-tokens 32768
```

### Batch (directory)

```bash
python inference/vLLM/batch_infer.py --image-dir /path/imgs --out-dir /path/out \
    --ports 8000 --task-type doc_parse --max-tokens 32768 --concurrency 16
```

- Each image produces a same-named `.md`; `out-dir/results.jsonl` records
  latency / char count / early-stop / post-processing. Completed items are
  skipped automatically.

---

## 5. Task types

`--task-type` selects the official recommended prompt. List them all:
`python inference/vLLM/infer_vllm_client.py --list-tasks`

| task_type          | Description                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| `doc_parse`        | End-to-end document parsing (default; body → md, tables → HTML, formulas → LaTeX, headers/footers ignored) |
| `structured_parse` | Structured parsing (ancient text / street view, etc.)                                                      |
| `spotting_json`    | Detection + recognition → JSON array (box normalized 0-1000 + text)                                        |
| `spotting_hunyuan` | Detection + recognition → Hunyuan coordinate format                                                        |
| `layout`           | Layout analysis                                                                                            |
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
inference/DFlash/
└── serve_DFlash.sh         # single-GPU vLLM + DFlash launch script (adds --speculative-config over AR)

inference/vLLM/             # shared with the AR path (see archive/vLLM.md)
├── serve.sh                # single-GPU vLLM AR launch script
├── infer_vllm_client.py    # single-image client   ┐ shared logic with Setup A/C;
└── batch_infer.py          # batch inference       ┘ outputs comparable
```

> The DFlash draft (config + dflash.py + tokenizer + model.safetensors) is not
> committed to Git; it comes from the `dflash/` subfolder of the HF model repo
> and is downloaded together with the base model (see §2).

> Shared helpers (`tasks.py` = task_type → prompt, `output_utils.py` =
> output utils incl. doc_parse normalization) live in a single copy at
> `inference/utils/` and are imported by all three setups (A/B/C).

> The `vision_config.max_image_size` in `config.json` is the positional-encoding
> table shape (a model-structure parameter) and **must not** be treated as a
> resolution knob.
