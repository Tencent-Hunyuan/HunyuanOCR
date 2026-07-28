# HunyuanOCR-1.5 · verl RL Training Kit <!-- omit in toc -->

> 🌏 **中文版**：[README.md](./README.md)

This directory (`train_verl/`) provides the complete RL training kit for HunyuanOCR-1.5: the GRPO training script, the reward scoring system, data preparation, Ray cluster launchers, and a [verl](https://github.com/volcengine/verl) fork adapted for HunyuanOCR-1.5. The goal here is to **make HunyuanOCR-1.5 RL training minimally runnable**, not to fully reproduce the experiments in the tech report; only the scaffolding and key adaptations needed to actually launch training are provided.

The underlying RL framework is verl (FSDP2 actor + vLLM async rollout). The upstream baseline is verl official `main` branch commit [`2b47a68`](https://github.com/volcengine/verl/commit/2b47a68b66d6fa21884990a5d4445ceba59c47e5) (2026-07-24 snapshot). All HunyuanOCR-1.5 adaptations live in the fork [`pspdada/verl-HYOCR` · branch `HYOCR`](https://github.com/pspdada/verl-HYOCR/tree/HYOCR); see [ADAPTATION_en.md](./ADAPTATION_en.md) for the detailed breakdown.

---

## Table of Contents <!-- omit in toc -->

- [1. Directory Layout](#1-directory-layout)
- [2. Environment Setup](#2-environment-setup)
- [3. Judge Server Deployment](#3-judge-server-deployment)
- [4. Ray Cluster](#4-ray-cluster)
- [5. Launching Training](#5-launching-training)
- [6. Reward System](#6-reward-system)
- [7. Checkpoints & Export](#7-checkpoints--export)
- [8. RLLoggingBoard (optional)](#8-rlloggingboard-optional)
- [9. Common Pitfalls](#9-common-pitfalls)
- [10. Related Documents](#10-related-documents)
- [11. Citation](#11-citation)

---

## 1. Directory Layout

The top level of `train_verl/` splits into four parts: training scripts + environment install, data preparation, the reward scoring system, and cluster / checkpoint helper scripts.

```
train_verl/
├── train_grpo.sh          # main entry: GRPO training launcher
├── install_cu13.sh        # CUDA 13 one-shot install (pip layer)
├── reward_ocr.py          # verl reward function entry (compute_score)
├── ADAPTATION_en.md       # verl adaptation notes
├── data/                  # raw JSONL → verl parquet
├── reward/                # reward scoring system (dispatcher + per-task rule scorers + judge client)
└── utils/                 # Ray cluster launchers + FSDP checkpoint merge helpers
```

The `verl-HYOCR` fork lives in its own repo; clone it locally into a `verl/` directory and `pip install -e .` so the code in this directory can import it (see [Section 2](#2-environment-setup)).

<details>
<summary>Expand the full directory tree</summary>

```
train_verl/
├── train_grpo.sh                       # main entry: GRPO training launcher
├── install_cu13.sh                     # CUDA 13 one-shot install (pip layer)
├── reward_ocr.py                       # verl reward function entry (compute_score)
├── ADAPTATION_en.md                    # verl adaptation notes
│
├── data/
│   └── prepare_data.py                 # raw JSONL → verl parquet
│
├── reward/                             # reward scoring system
│   ├── ocr_scorer.py                   # dispatcher + judge client
│   ├── ocr_utils.py                    # TaskType enum + generic helpers
│   ├── task_scorers/                   # per-task rule scorers
│   │   ├── spotting.py                 # text spotting
│   │   ├── layout.py                   # layout analysis
│   │   ├── parsing.py                  # generic parsing
│   │   ├── text_metrics.py             # shared text utilities across scorers
│   │   ├── ie/eval.py                  # information extraction (JSON field-level + parsing fallback)
│   │   └── chart_deplot/               # chart deplot (csv / mermaid / md-list)
│   └── utils/                          # generic helpers (prompt / request / server / judge_server_routes.json)
│
├── utils/
│   ├── ckpt/
│   │   └── merge_fsdp_ckpt_to_hf.sh    # FSDP2 shard → HuggingFace weights (optional)
│   └── ray/
│       ├── start_ray.sh                # bring up a Ray cluster (single-node / multi-node)
│       ├── stop_ray.sh
│       └── network_envs.sh             # NCCL / IB / GLOO env var template
```

</details>

---

## 2. Environment Setup

### 2.1 Hardware Recommendations <!-- omit in toc -->

- GPU: NVIDIA Hopper or Ampere (H20 / H800 / A100 etc.), 8 cards per node minimum, InfiniBand for multi-node.
- VRAM: ≥ 80 GB per card (HunyuanOCR-1.5 packs micro-batches by real token count under FSDP2 + vLLM async rollout).
- Disk: each training step produces sizable checkpoints; mount the output directory on high-throughput NVMe or a shared distributed filesystem.

### 2.2 Host Prerequisites <!-- omit in toc -->

Before installing Python packages, the host (or docker image) must already have:

- NVIDIA Driver ≥ 535.161.08
- CUDA ≥ 13.0 (13.3 recommended)
- Python 3.12.11 (we create a conda env below)
- System-level cuDNN, NCCL, nsight, ffmpeg, etc., provided by the host or image

### 2.3 Create the conda Environment <!-- omit in toc -->

```bash
conda create -n verl-hyocr python==3.12.11 -y
conda activate verl-hyocr
```

Every command below is expected to run inside the `verl-hyocr` env.

### 2.4 Install Dependencies <!-- omit in toc -->

```bash
cd train_verl
USE_MEGATRON=0 USE_SGLANG=0 bash install_cu13.sh
```

### 2.5 Install the verl-HYOCR fork <!-- omit in toc -->

```bash
git clone -b HYOCR https://github.com/pspdada/verl-HYOCR.git verl
cd verl
pip install -e . --no-deps
cd ..
```

The fork is named `verl-HYOCR` on GitHub, but the training scripts import it as `python -m verl.trainer.main_ppo`, so the local directory must be called `verl`. `git clone <url> verl` clones directly into a directory named `verl`. The paths listed in ADAPTATION.md (e.g. `verl/experimental/...`) then correspond one-to-one with your local layout.

### 2.6 Prepare Data <!-- omit in toc -->

`data/prepare_data.py` converts raw JSONL into a verl-compatible parquet dataset. Each JSONL line takes the following fields.

Required fields:

- `image` or `images`: image path (str for a single image, list for multi-image).
- `ref_answer`: the reference answer.
- `prompt`: a plain text prompt, or a full chat-messages list. When omitted, the script prepends one `<image>` placeholder per image in a single user turn.
- `question`: the user question at rollout time. vqa / translation splice it into the judge prompt; rule-based tasks may pass an empty string, but the field itself must be present.
- `task_type`: the key field that drives reward dispatch. Values: `Spotting` / `Layout` / `Parsing` / `IE` / `chart_deplot` / `Translation` / `VQA` (or the Chinese aliases `检测识别` / `版面分析` / `通用解析` / `信息抽取` / `图表解析` / `翻译` / `视觉问答`). If a file contains a single task, you may omit this field per record and pass `--task-type` on the CLI to stamp all rows uniformly.

Optional fields:

- `source_lang_text` / `target_lang`: translation only.

Output parquet schema (per row):

```python
{
    "data_source":  str,        # reward-router key; defaults to task_type
    "prompt":       list[dict], # chat messages; content is str with <image> placeholders
    "images":       list[str],  # absolute image paths, count matches <image> count
    "reward_model": {"style": "rule", "ground_truth": str},
    "extra_info": {
        "task_type":        str,        # required; drives reward dispatch
        "question":         str,        # optional; used by vqa / translation
        "img_path":         list[str],  # optional; kept so RLLoggingBoard can display it
        "source_lang_text": str,        # optional; translation only
        "target_lang":      str,        # optional; translation only
    },
}
```

Typical invocation:

```bash
python data/prepare_data.py \
    --inputs /path/to/a.jsonl /path/to/b.jsonl \
    --output_train /path/to/train.parquet \
    --output_val   /path/to/val.parquet \
    --val_ratio 0.02
```

Run `python data/prepare_data.py -h` for the full argument list.

---

## 3. Judge Server Deployment

Translation and VQA tasks are scored by a **judge model**; the remaining five task types use local rule scoring and do not need a judge. If your training data contains translation / vqa samples, you need an extra judge server running.

Start an OpenAI-compatible server with vLLM (Qwen/Qwen3-30B-A3B as an example):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 --max-model-len 16384
```

Once the server is up, register its address in `reward/utils/judge_server_routes.json`:

```json
{
  "Qwen/Qwen3-30B-A3B": ["10.0.0.1:8000", "10.0.0.2:8000"]
}
```

Multiple replicas are supported; the reward side does round-robin load balancing across them. If you keep the default judge model, leave `RM_SYSTEM_JUDGE_MODEL_NAME` in `train_grpo.sh` alone; if you swap in another model, update both `RM_SYSTEM_JUDGE_MODEL_NAME` and the key in `judge_server_routes.json` to match.

---

## 4. Ray Cluster

verl training depends on a Ray cluster that is brought up in advance; the driver attaches through `RAY_ADDRESS=auto`. Decoupling the cluster from the training process keeps the dashboard and slave nodes alive across driver restarts.

### 4.1 Single Node <!-- omit in toc -->

```bash
bash utils/ray/start_ray.sh local
```

### 4.2 Multi-Node <!-- omit in toc -->

```bash
# head node (INDEX=0)
NNODES=<N> INDEX=0 bash utils/ray/start_ray.sh <N> 0

# other nodes
NNODES=<N> INDEX=<i> bash utils/ray/start_ray.sh <N> <i>
```

`start_ray.sh` internally `source`s `utils/ray/network_envs.sh`, which exports NCCL / IB / GLOO variables tuned to your cluster. Multi-node runs typically need the following adjusted per `ibdev2netdev` and `ip -o link show` output:

- `NCCL_IB_HCA`
- `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`
- `UCX_NET_DEVICES`

If IB or GDR has compatibility problems, you can step down through three tiers A → B → C (IB+GDR → IB without GDR → plain TCP with `DISABLE_IB=1`); the top of `network_envs.sh` explains each tier.

Shut it down:

```bash
bash utils/ray/stop_ray.sh
```

---

## 5. Launching Training

One-liner:

```bash
bash train_grpo.sh
```

`train_grpo.sh` attaches to the Ray cluster, registers the reward function, wires up FSDP2 actor + vLLM rollout, and initialises the run directory.

### 5.1 Key Environment Variables <!-- omit in toc -->

The most commonly tweaked ones, overridable via env:

| Variable                     | Default                     | Meaning                                                          |
| :--------------------------- | :-------------------------- | :--------------------------------------------------------------- |
| `MODEL_PATH`                 | `/path/to/HunyuanOCR/model` | HunyuanOCR-1.5 weights directory (HF format)                     |
| `TRAIN_FILES`                | `/path/to/train.parquet`    | training parquet                                                 |
| `VAL_FILES`                  | `/path/to/val.parquet`      | validation parquet                                               |
| `OUTPUT_DIR`                 | `./outputs`                 | root directory for training artefacts                            |
| `NNODES`                     | `2`                         | number of nodes                                                  |
| `NGPUS_PER_NODE`             | `8`                         | GPUs per node                                                    |
| `PROJECT_NAME`               | `hyocr_1_5_verl`            | project name (organises the output sub-directory)                |
| `EXPERIMENT_NAME`            | `hyocr_1_5_grpo`            | experiment name                                                  |
| `REWARD_FN_PATH`             | `./reward_ocr.py`           | path to the reward function file                                 |
| `RM_SYSTEM_JUDGE_MODEL_NAME` | `Qwen/Qwen3-30B-A3B`        | judge model name; must match a key in `judge_server_routes.json` |

Single-node 8-GPU example:

```bash
NNODES=1 NGPUS_PER_NODE=8 \
MODEL_PATH=/data/hyocr_1_5 \
TRAIN_FILES=/data/train.parquet \
VAL_FILES=/data/val.parquet \
OUTPUT_DIR=/data/outputs \
bash train_grpo.sh
```

Multi-node example (start Ray on every node first, then run on the head node):

```bash
NNODES=4 NGPUS_PER_NODE=8 \
MODEL_PATH=/data/hyocr_1_5 \
TRAIN_FILES=/data/train.parquet \
VAL_FILES=/data/val.parquet \
OUTPUT_DIR=/data/outputs \
bash train_grpo.sh
```

Finer-grained knobs (batch size, KL coefficient, rollout sampling params, logging cadence, …) are documented as inline comments in `train_grpo.sh`; edit the script directly rather than pushing every switch through env vars.

### 5.2 Run Artefacts <!-- omit in toc -->

Once training starts, `OUTPUT_DIR/${PROJECT_NAME}/${EXPERIMENT_NAME}/` gets the following layout:

```
├── ckpt/                       # verl-saved FSDP2 sharded checkpoints
├── tensorboard/                # TensorBoard event files
├── metrics.jsonl               # per-step metrics (VERL_FILE_LOGGER_PATH)
├── stdout.log                  # training stdout
└── rl_logging_board/           # RLLoggingBoard per-sample JSONL dump (see Section 8)
```

---

## 6. Reward System

### 6.1 Data Flow <!-- omit in toc -->

verl's `NaiveRewardManager` calls the reward function once per rollout:

```
compute_score(data_source, solution_str, ground_truth, extra_info)
    → OCRScorer.process_scoring({...})
        → process_<task>_task(response, ref_answer) in task_scorers/<xxx>.py
```

- `reward_ocr.py`: the verl-side reward function entry; maps kwargs into the internal scoring_obj.
- `reward/ocr_scorer.py`: `OCRScorer.process_scoring` is the dispatcher, branching by `task_type`.
- `reward/task_scorers/`: one entry function per task, all named `process_<task>_task(response, ref_answer) -> dict`.

### 6.2 Scoring Policy per Task Type <!-- omit in toc -->

| Task         | Dispatch condition (`TaskType.is_xxx`) | Scoring method                                                                                 |
| :----------- | :------------------------------------- | :--------------------------------------------------------------------------------------------- |
| spotting     | `"Spotting"` / `"检测识别"`            | Rule: format detection + IoU matching + normalised edit distance                               |
| layout       | `"Layout"` / `"版面分析"`              | Rule: hy-meta / JSON dual format, F1 mode or edit-distance mode                                |
| parsing      | `"Parsing"` / `"通用解析"`             | Rule: character accuracy + TEDS table score                                                    |
| ie           | `"IE"` / `"信息抽取"`                  | Rule: JSON field-level exact_match + edit-distance; falls back to parsing when ref is non-JSON |
| chart_deplot | `"chart_deplot"` / `"图表解析"`        | Rule: dispatch by ref format to csv / tree / flowchart evaluation                              |
| translation  | `"Translation"` / `"翻译"`             | Judge model: 0-5 score mapped to `[0, 1]` by piecewise linear transform                        |
| vqa          | `"VQA"` / `"视觉问答"`                 | Judge model: judgement 0 / 1                                                                   |

Matching uses substring containment (English alias with case-insensitive match, Chinese alias verbatim), so any task_type string that contains the relevant keyword hits the right branch.

> **Relation to the inference-side `task_type`**: the 12 `task_type` keys in `inference/utils/tasks.py` (`doc_parse` / `spotting_json` / `layout_parse` / ...) are **inference-time prompt variants** selected via `--task-type` to pick the official prompt sent to the model. The 7 training categories here are **RL reward dispatch routes**. The two vocabularies live at different abstraction levels ("which instruction to send" vs. "which reward rule to apply") and are not interchangeable.

### 6.3 Adding a New Task <!-- omit in toc -->

To add a new rule-based task, follow three steps:

1. In `reward/ocr_utils.py`, add an English + Chinese member to the `TaskType` enum (e.g. `MY_TASK = "MyTask"` / `MY_TASK_ZH = "我的任务"`) and one classifier `is_my_task` (a one-line dispatch to `_matches`).
2. Under `reward/task_scorers/`, add `my_task.py` implementing `process_my_task_task(response, ref_answer) -> dict` (return `{"analysis", "is_valid", "reward"}`); wrap it in try/except that folds errors into `reward=-1.0` so nothing raises to the dispatcher.
3. In `reward/ocr_scorer.py`, add a three-line branch in `process_scoring`; re-export the new entry in `reward/task_scorers/__init__.py`.

Judge-model tasks take one extra step: register system / template strings in `JUDGE_PROMPTS` under `reward/utils/prompt.py` and add the corresponding score-extraction regex in `validate_format_response`.

---

## 7. Checkpoints & Export

### 7.1 Training Artefacts <!-- omit in toc -->

Checkpoints saved by verl live in `OUTPUT_DIR/${PROJECT_NAME}/${EXPERIMENT_NAME}/ckpt/global_step_${N}/`:

```
global_step_${N}/actor/
├── model_world_size_${WS}_rank_*.pt   # per-rank FSDP2 shards
├── optim_world_size_${WS}_rank_*.pt
├── extra_state_world_size_${WS}_rank_*.pt
├── fsdp_config.json
└── huggingface/                        # config / tokenizer / processor / chat_template
```

### 7.2 Merge into HuggingFace Weights (optional) <!-- omit in toc -->

If you need to merge FSDP2 sharded checkpoints into standard HF safetensors (e.g. for offline evaluation or release), run:

```bash
# merge a single step
STEPS=100 bash utils/ckpt/merge_fsdp_ckpt_to_hf.sh

# batch
STEPS="25 50 75 100" bash utils/ckpt/merge_fsdp_ckpt_to_hf.sh

# merge every step under CKPT_ROOT
STEPS=all bash utils/ckpt/merge_fsdp_ckpt_to_hf.sh
```

Internally this calls `python -m verl.model_merger merge --backend fsdp ...`, identical to what training produces when `actor.checkpoint.save_contents` includes `'hf_model'`. Default output is written in-place under `actor/huggingface/`.

---

## 8. RLLoggingBoard (optional)

`train_grpo.sh` enables the dual-sink `_log_rollout_data` in verl: alongside the built-in `{step}.jsonl`, each training step also dumps rollout data as per-sample JSONL in the format expected by the [RLLoggingBoard](https://github.com/HarderThenHarder/RLLoggingBoard) UI (fields include prompt / response / reward / logprobs / advantages / kl / img_path / data_source / task_type / …). The dumps land under `OUTPUT_DIR/${PROJECT_NAME}/${EXPERIMENT_NAME}/rl_logging_board/`.

Install RLLoggingBoard and point it at that directory to open the visual dashboard: per-sample rollout quality, reward distribution, KL curves, and so on.

---

## 9. Common Pitfalls

- **Keep `use_fast_processor=False` default.** The current transformers version's `HunYuanVLImageProcessorFast` (torchvision backend) has known issues; the fix in [PR #47499](https://github.com/huggingface/transformers/pull/47499) is merged but not yet released. Stay on the slow (PIL) backend for now.
- **Keep `use_remove_padding=False` default.** The current transformers version does not natively support sequence packing / remove padding for the high-performance forward path; flipping to True raises or gives wrong results.
- **`bad_words` + `add_vision_logit_bias` are a double safety net.** The training script adds six vision control tokens to `bad_words` and simultaneously turns on `add_vision_logit_bias=True` to push a `-1e9` logit bias before sampling. If any of these tokens leak into the generated sequence, they break both reward scoring and the actor forward's image branch, so keep both switches on together.
- **Single-node runs work out of the box; multi-node needs NCCL env vars exported.** `network_envs.sh` only forwards NCCL / IB / GLOO variables that the driver has already exported and does nothing otherwise. Single-node defaults to NVLink / local socket; multi-node runs need `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` / etc. configured per `ibdev2netdev` output. See [ADAPTATION_en.md B7](./ADAPTATION_en.md#b7-nccl--gloo-network-variable-forwarding-to-ray-workers) for the 25-entry whitelist.

---

## 10. Related Documents

- [ADAPTATION_en.md](./ADAPTATION_en.md): verl adaptation notes, grouped into A (HunyuanOCR-specific RL adaptations) / B (general functional adaptations) / C (mandatory hyperparameters).
- [verl-HYOCR fork](https://github.com/pspdada/verl-HYOCR/tree/HYOCR): the fork's full source.
- [verl upstream](https://github.com/volcengine/verl): upstream repo, useful for diffing.
- [RLLoggingBoard](https://github.com/HarderThenHarder/RLLoggingBoard): training-log visualisation UI.

---

## 11. Citation

```bibtex
@article{HunyuanOCR_1_5_2026,
  title   = {{HunyuanOCR-1.5}: Making Lightweight {OCR} {VLMs} Faster and Better},
  author  = {Li, Gengluo and Wan, Xingyu and Peng, Shangpin and Wang, Weinong and Feng, Hao and Du, Yongkun and Wu, Binghong and Ruan, Zheng and Lu, Zhiqiong and Wu, Liang and Lyu, Pengyuan and Shen, Huawen and Lin, Zibin and Hu, Shijing and Yang, Jieneng and Wen, Hongbing and Yu, Guanghua and Liu, Hong and Wang, Bochao and Ma, Can and Hu, Han and Zhang, Chengquan and Zhou, Yu},
  journal = {arXiv preprint arXiv:2607.04884},
  year    = {2026}
}
```
