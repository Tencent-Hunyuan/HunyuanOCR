# Data Format & Packing Pipeline

[中文阅读](./data_format_zh.md)

## Overview

Training expects **packed** JSONL: each line contains multiple original samples concatenated into a single sequence up to `pack_length` tokens. This maximizes GPU utilization by removing padding waste.

The pipeline: raw JSONL → tokenize + count → pack → training-ready JSONL.

---

## 1. Raw OCR JSONL Schema

Each line in a raw JSONL file is one training sample. The packing script's `normalize_sample()`
accepts two input formats, resolved in the following priority order.

### Format A (preferred): `conv` + `img_path_sh` / `img_path_cq`

```json
{
  "img_path_sh": "/absolute/path/to/image.png",
  "conv": [
    { "question": "Extract all body text from the document image as markdown...", "answer": "# Title\n\nBody text ..." }
  ]
}
```

| Field                       | Type         | Required | Notes                                                                      |
| --------------------------- | ------------ | -------- | -------------------------------------------------------------------------- |
| `img_path_sh`/`img_path_cq` | `str`        | ✅       | Absolute image path; `img_path_sh` is used if present, else `img_path_cq`. |
| `conv`                      | `list[dict]` | ✅       | The 0-th turn's `question` / `answer` is used.                             |

### Format B (fallback): `image_path` + `conversations`

```json
{
  "image_path": ["/absolute/path/to/image.png"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nExtract all body text from the document image as markdown..."
    },
    { "from": "gpt", "value": "# Title\n\nBody text ..." }
  ]
}
```

| Field           | Type         | Required | Notes                                                                                |
| --------------- | ------------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `image_path`    | `list[str]`  | ✅       | List of absolute paths; the first image is used.                |
| `conversations` | `list[dict]` | ✅       | Alternating `human`/`gpt` (or `user`/`assistant`) turns. The `<image>` placeholder in the `human` value is stripped and the remainder becomes the question. |

> Format B is only used when the sample has no `conv` field. Both formats are normalized to an
> `image` / `question` / `answer` triple.

## 2. Packing Pipeline

The `tools/pipeline_count_and_pack.py` script does two things:

1. **Count phase** (parallel): tokenizes each sample and writes token counts to `count_output_dir/*.jsonl`
2. **Pack phase**: greedily packs samples up to `pack_length` tokens per output line (First-Fit Decreasing)

### Config file: `configs/data_list.txt`

Plain text file, one path per line:

```
/path/to/dataset_A.jsonl
/path/to/dataset_B.jsonl
/path/to/dataset_C.jsonl
# Comments starting with # are ignored
```

### Run

```bash
MODEL_PATH=/path/to/HunyuanOCR \
INPUT_LIST=./configs/data_list.txt \
PACK_OUTPUT=./data/parsing_packed_20480.jsonl \
    bash scripts/pack_data.sh
```

### Tunable parameters (env vars)

| Env var               |                                  Default | Description                                           |
| --------------------- | ---------------------------------------: | ----------------------------------------------------- |
| `MODEL_PATH`          |                             _(required)_ | HunyuanOCR base model dir (for tokenizer + processor) |
| `INPUT_LIST`          |                `./configs/data_list.txt` | Path to file listing raw JSONLs                       |
| `COUNT_OUTPUT_DIR`    |             `./data/parsing_jsonl_count` | Temp dir for count-phase output                       |
| `PACK_OUTPUT`         | `./data/parsing_packed_{PACK_LEN}.jsonl` | Final packed JSONL                                    |
| `PACK_LEN`            |                                  `20480` | Max sequence length per packed line                   |
| `NUM_PROCESSES`       |                                     `32` | Multiprocess count workers                            |
| `THREADS_PER_PROCESS` |                                      `8` | Threads per count worker                              |
| `LOG_FILE`            |                          `pack_data.log` | Progress log path                                     |

### Output schema (packed JSONL)

Each output line is a JSON array. Each item is the sample consumed by the training dataset:

```json
[
    {
        "image": "/abs/path/1.png",
        "question": "Extract the text from the image",
        "answer": "Recognized document text",
        "num_tokens": 4123
    },
    {
        "image": "/abs/path/2.png",
        "question": "Extract the text from the image",
        "answer": "Recognized document text",
        "num_tokens": 4444
    }
]
```

The sum of `num_tokens` in one array is at most `pack_length`. `cu_seqlens` is recomputed by the training collator and is not stored in the JSONL.

### Performance tips

- CPU-bound: scales linearly with `NUM_PROCESSES`. On 128-core machine, use `NUM_PROCESSES=32~64`.
- Image data does **not** need to be pre-loaded — the count phase only tokenizes text; images are loaded lazily during training.
- First run generates count cache; re-running with the same input list skips the count phase.

## 3. Verifying Packed Data

Sanity check the packed JSONL:

```bash
python -c "
import json
with open('./data/parsing_packed_20480.jsonl') as f:
    for i, line in enumerate(f):
        pack = json.loads(line)
        print(f'pack {i}: {len(pack)} samples, '
              f'{sum(item.get(\"num_tokens\", 0) for item in pack)} tokens')
        if i >= 3: break
"
```

You should see something like:

```
pack 0: 7 samples, 20438 tokens
pack 1: 5 samples, 19821 tokens
pack 2: 12 samples, 20301 tokens
pack 3: 3 samples, 18654 tokens
```

Each pack is close to `pack_length=20480` — that's the goal.
