#!/bin/bash
# ============================================================================
# SFT the HunyuanOCR base model on a SINGLE 80GB GPU (A100-80 / H100).
#
# Same entry point as scripts/sft_base.sh, retuned for one GPU and for a small
# domain dataset (~1k pages) instead of a cluster and ~1M packs:
#
#   NPROC_PER_NODE  8    -> 1
#   env_common.sh        -> env_single.sh   (no InfiniBand on a single box)
#   packed_max_length    20480 -> 16384     (upstream notes 20480 OOMs at 80GB)
#   GRAD_ACCUM      1    -> 4               (1 pack/step is too noisy a batch)
#   SAVE_STEPS      200  -> 50              (a whole epoch is well under 200)
#
# Everything is still env-overridable, e.g.  PACK_LEN=20480 EPOCHS=3 bash ...
# ============================================================================

set -e

# ────────────── Common env ──────────────
source scripts/env_single.sh

# ────────────── Distributed configuration ──────────────
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NODE_RANK=${NODE_RANK:-0}

# ────────────── Model & data paths ──────────────
model_name_or_path=${MODEL_PATH:?set MODEL_PATH to the HunyuanOCR base model dir}
train_data_path=${TRAIN_DATA:-./data/packed/train_16384.jsonl}
image_path="not_needed"    # packed records carry absolute image paths

# ────────────── Hyperparameters ──────────────
lr=${LR:-2e-5}
batch_size=${BATCH_SIZE:-1}
grad_accum_steps=${GRAD_ACCUM:-4}
num_epochs=${EPOCHS:-5}
warmup_ratio=${WARMUP:-0.03}
save_steps=${SAVE_STEPS:-50}
pack_len=${PACK_LEN:-16384}

# Which parts to train. Freezing the vision tower (TUNE_VISION=False) saves
# roughly 15% of step time and is worth trying if the pages are clean scans.
tune_vision=${TUNE_VISION:-True}
tune_mlp=${TUNE_MLP:-True}
tune_llm=${TUNE_LLM:-True}

# Optional ZeRO-2. Not needed for a 1B model at 80GB; set DEEPSPEED=scripts/zero2.json
# only if you hit OOM after lowering PACK_LEN.
deepspeed_arg=""
[ -n "${DEEPSPEED}" ] && deepspeed_arg="--deepspeed ${DEEPSPEED}"

entry_file=train/train_hunyuan.py

# ────────────── Output ──────────────
run_name=${RUN_NAME:-"hyocr_sft_1gpu_lr${lr}_ep${num_epochs}_$(date +%m%d_%H%M)"}
output_dir=${OUTPUT_DIR:-"./output/${run_name}"}
TENSORBOARD_DIR="${output_dir}/tensorboard/$(date "+%Y.%m.%d-%H.%M.%S")"
mkdir -p "${TENSORBOARD_DIR}"

echo "========================================"
echo "Run name        : ${run_name}"
echo "Output dir      : ${output_dir}"
echo "Base model      : ${model_name_or_path}"
echo "Train data      : ${train_data_path}"
echo "----- Hyperparameters -----"
echo "  gpus          : ${NPROC_PER_NODE}"
echo "  lr            : ${lr}"
echo "  epochs        : ${num_epochs}"
echo "  batch x accum : ${batch_size} x ${grad_accum_steps}"
echo "  pack length   : ${pack_len}"
echo "  tune v/m/l    : ${tune_vision} / ${tune_mlp} / ${tune_llm}"
echo "  deepspeed     : ${DEEPSPEED:-off}"
echo "========================================"

# ────────────── Training args ──────────────
args="
    --model_name_or_path ${model_name_or_path} \
    --train_data_path ${train_data_path} \
    --image_folder ${image_path} \
    --data_flatten True \
    --data_packing True \
    --tune_mm_vision ${tune_vision} \
    --tune_mm_mlp ${tune_mlp} \
    --tune_mm_llm ${tune_llm} \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --logging_steps 5 \
    --packed_max_length ${pack_len} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --logging_dir ${TENSORBOARD_DIR} \
    --report_to tensorboard ${deepspeed_arg}"

# ────────────── Launch ──────────────
torchrun --nproc_per_node="${NPROC_PER_NODE}" \
         --master_addr="${MASTER_ADDR}" \
         --master_port="${MASTER_PORT}" \
         --node_rank="${NODE_RANK}" \
         --nnodes="${NNODES}" \
         ${entry_file} ${args} 2>&1 | tee "${output_dir}/train_${NODE_RANK}.log"
