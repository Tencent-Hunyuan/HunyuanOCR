#!/bin/bash
# ============================================================================
# DFlash draft model — continue-finetune from an EXISTING DFlash checkpoint.
#
# Use this when you have a pre-trained DFlash draft (e.g. our released v1) and
# want to adapt it to a smaller / domain-specific OCR dataset.
#
# Recommended profile:
#   - lr        = 2e-5  (5× smaller than from-scratch; finetune LR)
#   - epochs    = 10    (small data → more passes needed)
#   - warmup    = 0.05  (slightly longer warmup for stability)
#   - save_steps = 500  (~10 ckpts over full run)
#
# Empirical result on 14.7k packs: this continue-finetune profile beats the
# from-scratch baseline (1M packs) on both acceptance rate (42% vs 33%) and
# end-to-end speedup (2.14× vs 1.92×).
# ============================================================================

set -e

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

# ────────────── Model paths ──────────────
# (a) BASE model: frozen target backbone
model_name_or_path=${MODEL_PATH:-/path/to/HunyuanOCR/base/model}

# (b) DFlash init dir: existing draft checkpoint to continue-train from.
#     Must contain config.json + model.safetensors (or pytorch_model.bin).
#     The draft config is loaded from THIS directory when it is a valid dir;
#     otherwise the code falls back to HYOCR_DFLASH_CONFIG_DIR (below).
dflash_init_dir=${DFLASH_INIT:-/path/to/existing/dflash/checkpoint}

# (c) Fallback draft-config template (used only when dflash_init_dir is not a
#     valid directory). Default: train/configs/ (bundled with the repo).
export HYOCR_DFLASH_CONFIG_DIR=${HYOCR_DFLASH_CONFIG_DIR:-train/configs}

# (d) Training data
train_data_path=${TRAIN_DATA:-./data/parsing_packed_20480.jsonl}
image_path="not_needed"

# ────────────── Hyperparameters (v3 finetune profile) ──────────────
lr=${LR:-2e-5}
batch_size=${BATCH_SIZE:-1}
grad_accum_steps=${GRAD_ACCUM:-1}
num_epochs=${EPOCHS:-10}
warmup_ratio=${WARMUP:-0.05}
save_steps=${SAVE_STEPS:-500}

# DFlash-specific
num_mask_tokens=${NUM_MASK_TOKENS:-16}
loop_num=${LOOP_NUM:-1}
sample_block_num=${SAMPLE_BLOCK_NUM:-8}

entry_file=train/train_draft_from_dflash.py

# ────────────── Output ──────────────
run_name="hyocr_dflash_ft_lr${lr}_ep${num_epochs}_$(date +%m%d_%H%M)"
output_dir="./output/${run_name}"
TENSORBOARD_DIR="${output_dir}/tensorboard/$(date "+%Y.%m.%d-%H.%M.%S")"
mkdir -p "${TENSORBOARD_DIR}"

echo "========================================"
echo "Run name        : ${run_name}"
echo "Output dir      : ${output_dir}"
echo "DFlash init dir : ${dflash_init_dir}"
echo "Base model      : ${model_name_or_path}"
echo "Train data      : ${train_data_path}"
echo "----- Hyperparameters -----"
echo "  lr            : ${lr}"
echo "  epochs        : ${num_epochs}"
echo "  warmup_ratio  : ${warmup_ratio}"
echo "  save_steps    : ${save_steps}"
echo "  batch_size    : ${batch_size}"
echo "  grad_accum    : ${grad_accum_steps}"
echo "  num_mask_toks : ${num_mask_tokens}"
echo "  sample_blocks : ${sample_block_num}"
echo "========================================"

# ────────────── Training args ──────────────
args="
    --model_name_or_path ${model_name_or_path} \
    --dflash_init_dir ${dflash_init_dir} \
    --train_data_path ${train_data_path} \
    --image_folder ${image_path} \
    --data_flatten True \
    --data_packing True \
    --num_mask_tokens ${num_mask_tokens} \
    --use_kv_cache False \
    --loop_num ${loop_num} \
    --sample_block_num ${sample_block_num} \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --logging_steps 10 \
    --packed_max_length 20480 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --logging_dir ${TENSORBOARD_DIR} \
    --report_to tensorboard"

# ────────────── Launch ──────────────
echo "Launch training on NODE_RANK=${NODE_RANK}"
torchrun --nproc_per_node="${NPROC_PER_NODE}" \
         --master_addr="${MASTER_ADDR}" \
         --master_port="${MASTER_PORT}" \
         --node_rank="${NODE_RANK}" \
         --nnodes="${NNODES}" \
         ${entry_file} ${args} 2>&1 | tee "${output_dir}/train_${NODE_RANK}.log"
