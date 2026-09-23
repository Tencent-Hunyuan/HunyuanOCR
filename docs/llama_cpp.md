# PC-side deployment with llama.cpp

[中文阅读](./llama_cpp_zh.md)

HunyuanOCR-1.5 can be deployed on **CPU / consumer GPU / laptop** via
[`llama.cpp`](https://github.com/ggml-org/llama.cpp), converting the base model
(and optionally the DFlash draft) to the GGUF format and serving via
`llama-server` (OpenAI-compatible).

> ✅ **DFlash is now upstream.** HunyuanOCR DFlash speculative decoding was
> merged into `ggml-org/llama.cpp` by
> [PR #28890](https://github.com/ggml-org/llama.cpp/pull/28890)
> (commit [`828fdf2`](https://github.com/ggml-org/llama.cpp/commit/828fdf282e195300c2965bd9511807e24ed53bdb),
> first shipped in build
> [`b11103`](https://github.com/ggml-org/llama.cpp/releases/tag/b11103)).
> The previous DFlash fork is no longer needed — use upstream `master` (or any
> build `>= b11103`) for both the base model and DFlash.

If you followed the earlier version of this guide, see
[§5 Migrating from the old fork](#5-migrating-from-the-old-fork) for the flag
renames.

---

## 1. Clone & build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Add -DGGML_CUDA=ON if you have an NVIDIA GPU and want CUDA acceleration.
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON
cmake --build ./build --config Release -j
```

Verify that your checkout is new enough to know about DFlash:

```bash
build/bin/llama-server --help | grep -A2 -- --spec-type
# the listed types must include draft-dflash
```

## 2. Set up a Python env for weight conversion

```bash
uv venv --python 3.12 venv-llamacpp
source venv-llamacpp/bin/activate
uv pip install huggingface_hub transformers torch openai
```

## 3. Download HunyuanOCR weights and convert to GGUF

```bash
hf download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

The HF repo ships the DFlash draft under the `dflash/` subfolder, so the single
download above pulls both the base model and the draft.

```bash
# Language / decoder weights → hyocr-f16.gguf
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR/hyocr-f16.gguf \
    --outtype f16 \
    ./HunyuanOCR

# Vision (mmproj) weights → mmproj-hyocr-f16.gguf
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR/mmproj-hyocr-f16.gguf \
    --outtype f16 \
    --mmproj \
    ./HunyuanOCR
```

### 3.1 Convert the DFlash draft weights (optional)

Skip this if you only want the base model. `--target-model-dir` points at the
base HunyuanOCR HF checkpoint (needed for tokenizer / hidden size / layer
count), while the positional argument points at the DFlash checkpoint
directory.

```bash
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR/hyocr-dflash-bf16.gguf \
    --outtype bf16 \
    --target-model-dir ./HunyuanOCR \
    ./HunyuanOCR/dflash
```

---

## 4. Launch the OpenAI-compatible server

### 4.1 Base model only

```bash
build/bin/llama-server \
    --model  "./HunyuanOCR/hyocr-f16.gguf" \
    --mmproj "./HunyuanOCR/mmproj-hyocr-f16.gguf" \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096 \
    -fa on --jinja
```

The endpoint is `http://<host>:8080/v1/chat/completions`, alias `HYVL`.

### 4.2 With DFlash speculative decoding

```bash
build/bin/llama-server \
    --model       "./HunyuanOCR/hyocr-f16.gguf" \
    --mmproj      "./HunyuanOCR/mmproj-hyocr-f16.gguf" \
    --spec-draft-model "./HunyuanOCR/hyocr-dflash-bf16.gguf" \
    --spec-type draft-dflash --spec-draft-n-max 15 \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096 \
    --parallel 1 \
    -fa on --jinja
```

Key DFlash-specific flags:

| Flag                                | Meaning                                                                                                                                                                  |
| :---------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--spec-draft-model <path>` (`-md`) | GGUF path of the DFlash draft model                                                                                                                                      |
| `--spec-type draft-dflash`          | Select DFlash speculative decoding. Auto-detected from the draft GGUF metadata; passing it explicitly keeps the choice unambiguous.                                      |
| `--spec-draft-n-max 15`             | Draft tokens per speculative step. The released draft has `block_size = 16`, so 15 is the maximum; larger values are clamped. Default is only 3, so set this explicitly. |
| `--parallel 1`                      | Single serial slot (recommended for DFlash on PC)                                                                                                                        |
| `-fa on`                            | Flash attention on both target and draft                                                                                                                                 |
| `--jinja`                           | Use the chat template shipped with the checkpoint                                                                                                                        |

For very high-resolution inputs you can additionally raise
`--batch-size / --ubatch-size` (we used `8192` on our own boxes) so a whole
image chunk is prefilled in one batch.

### 4.3 Measured DFlash speedup

From the upstream PR, on an Apple M5 Pro (6 performance + 12 efficiency cores,
48 GB unified memory), macOS 26.4.1, Metal backend, Release build, over the 26
document-image OCR requests shipped in this repo:

| Metric               | Value                                     |
| :------------------- | :---------------------------------------- |
| draft acceptance     | ~0.5                                      |
| mean accepted length | 8.6                                       |
| output               | byte-identical to the non-speculative run |
| decode               | 84.6 → 167.1 t/s (~2×)                    |
| prefill              | 232 → 192 t/s (~17% slower)               |

Prefill regresses because the draft also encodes the whole prompt, so
end-to-end is roughly break-even on short answers and clearly faster on long
structured outputs (dense documents, tables, formulas).

---

## 5. Migrating from the old fork

| Old (fork)             | New (upstream)                                                      |
| :--------------------- | :------------------------------------------------------------------ |
| `--model-draft <path>` | `--spec-draft-model <path>` (`-md`, `--model-draft` still accepted) |
| `--dflash`             | `--spec-type draft-dflash`                                          |
| `--draft-max 16`       | `--spec-draft-n-max 15`                                             |

The draft GGUF conversion command is unchanged. Note that the old
`--draft-max 16` exceeded the trained block size and was clamped to 15 anyway.

---

## 6. Quick verification

We ship a minimal OpenAI-compatible client and 26 test OCR images under
[`llama_cpp/`](../llama_cpp) so you can smoke-test the deployment end-to-end.

### 6.1 Install the client dep

```bash
pip install openai
```

### 6.2 Run

```bash
cd llama_cpp
python chat.py
```

By default `chat.py` targets `http://127.0.0.1:8080/v1` with alias `HYVL`
(matching the `llama-server` launch commands above), reads
`test_assets/data.jsonl`, sends the first `ocr` sample to the server, prints
the response and per-item elapsed time, and tees everything into
`logs/chat_<timestamp>.log`.

Tune the client behavior at the top of `chat.py`:

```python
BASE_URL     = "http://127.0.0.1:8080/v1"
MODEL        = "HYVL"
MAX_REQUESTS = 10                # cap total requests
TYPE_LIMITS  = {"ocr": 1}        # per-type cap; set None to disable
```

### 6.3 Example output

```
=== [ocr] ocr/0.png ===
Prompt: 请提取文档图片中正文的所有信息用 markdown 格式表示。
ring, and Jacobson semisimple, by Corollary 8.35(ii)]. The factor module
$ J^{q}/J^{q+1} $ is an $ (R/J) $-module; hence, by Corollary 8.43,
$ J^{q}/J^{q+1} $ is a semisimple module, and so it can be decomposed into a
direct sum of (possibly infinitely many) simple $ (R/J) $-modules. ...
...
[elapsed] 6.885s

[total] 1 items (ocr=1), elapsed: 6.885s
```

If you see the response streamed back and a `[total]` line at the end, the
llama.cpp deployment (± DFlash) is working correctly.

When DFlash is enabled, the server log also reports the draft acceptance rate,
which is the quickest way to confirm speculative decoding is actually running.
