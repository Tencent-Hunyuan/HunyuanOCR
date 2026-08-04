#!/bin/bash
# ============================================================================
# SFT training for the HunyuanOCR base model.
#
# Trains the full HunyuanOCR (vision encoder + MLP + LLM) end-to-end on packed
# OCR sequence data. Default profile:
#   - lr        = 2e-5 (small LR for SFT)
#   - epochs    = 5
#   - grad accum = 1, batch = 1 per GPU
#   - packed_max_length = 20480
#
# Multi-node: pass NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT as env vars.
# ============================================================================

set -euo pipefail

# Activate your Python environment before running:
#   source /path/to/miniconda3/bin/activate && conda activate hyocr

# ────────────── Common env ──────────────
source scripts/env_common.sh

# ────────────── Distributed configuration ──────────────
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}

echo "总节点数 (Total nodes) : $NNODES"
echo "主节点地址 (Master)     : $MASTER_ADDR"
echo "主节点端口 (Port)       : $MASTER_PORT"
echo "NODE_RANK              : $NODE_RANK"

# ────────────── Model & data paths ──────────────
model_name_or_path=${MODEL_PATH:-/path/to/HunyuanOCR/base/model}
train_data_path=${TRAIN_DATA:-./data/parsing_packed_20480.jsonl}
image_path="not_needed"    # images are embedded inside the packed data

# ────────────── Hyperparameters ──────────────
lr=${LR:-2e-5}
batch_size=${BATCH_SIZE:-1}
grad_accum_steps=${GRAD_ACCUM:-1}
num_epochs=${EPOCHS:-5}
save_steps=${SAVE_STEPS:-200}
data_flatten=${DATA_FLATTEN:-True}
data_packing=${DATA_PACKING:-True}
packed_max_length=${PACKED_MAX_LENGTH:-20480}
dataloader_workers=${DATALOADER_WORKERS:-4}
max_steps=${MAX_STEPS:--1}

entry_file=train/train_hunyuan.py

# ────────────── Output ──────────────
run_name="hyocr_sft_base_lr${lr}_ep${num_epochs}_$(date +%m%d_%H%M)"
output_dir="./output/${run_name}"
TENSORBOARD_DIR="${output_dir}/tensorboard/$(date "+%Y.%m.%d-%H.%M.%S")"
mkdir -p "${TENSORBOARD_DIR}"

echo "========================================"
echo "Run name        : ${run_name}"
echo "Output dir      : ${output_dir}"
echo "Base model      : ${model_name_or_path}"
echo "Train data      : ${train_data_path}"
echo "  lr            : ${lr}"
echo "  epochs        : ${num_epochs}"
echo "  batch_size    : ${batch_size}"
echo "  grad_accum    : ${grad_accum_steps}"
echo "========================================"

# ────────────── Training args ──────────────
args=(
    --model_name_or_path "$model_name_or_path"
    --train_data_path "$train_data_path"
    --image_folder "$image_path"
    --data_flatten "$data_flatten"
    --data_packing "$data_packing"
    --tune_mm_vision True
    --tune_mm_mlp True
    --tune_mm_llm True
    --bf16
    --output_dir "$output_dir"
    --num_train_epochs "$num_epochs"
    --max_steps "$max_steps"
    --per_device_train_batch_size "$batch_size"
    --per_device_eval_batch_size "$((batch_size * 2))"
    --gradient_accumulation_steps "$grad_accum_steps"
    --eval_strategy no
    --save_strategy steps
    --save_steps "$save_steps"
    --save_total_limit 3
    --learning_rate "$lr"
    --weight_decay 0.01
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type cosine_with_min_lr
    --logging_steps 1
    --packed_max_length "$packed_max_length"
    --gradient_checkpointing True
    --dataloader_num_workers "$dataloader_workers"
    --run_name "$run_name"
    --logging_dir "$TENSORBOARD_DIR"
    --report_to tensorboard
)

# ────────────── Launch ──────────────
echo "Launch training on NODE_RANK=${NODE_RANK}"
torchrun --nproc_per_node="${NPROC_PER_NODE}" \
         --master_addr="${MASTER_ADDR}" \
         --master_port="${MASTER_PORT}" \
         --node_rank="${NODE_RANK}" \
         --nnodes="${NNODES}" \
         "${entry_file}" "${args[@]}" 2>&1 | tee "${output_dir}/train_${NODE_RANK}.log"
