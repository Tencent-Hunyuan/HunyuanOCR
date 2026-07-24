#!/usr/bin/env bash
# =============================================================================
# merge_fsdp_ckpt_to_hf.sh
#
# 把 verl 保存的 FSDP2 sharded actor checkpoint 离线合并成 HuggingFace 标准权重
# （safetensors 分片 + config.json + tokenizer + processor + generation_config）
# 底层调用: `python -m verl.model_merger merge --backend fsdp ...`，
# 与训练时 `actor.checkpoint.save_contents` 包含 'hf_model' 得到的结果完全一致。
#
# 前置条件：
#   ${CKPT_ROOT}/global_step_${N}/actor/
#     ├── model_world_size_${WS}_rank_*.pt   ← 每 rank 的 FSDP2 shard
#     ├── optim_world_size_${WS}_rank_*.pt   （合并 hf 权重时用不到）
#     ├── extra_state_world_size_${WS}_rank_*.pt
#     ├── fsdp_config.json
#     └── huggingface/                        ← 已经有 config.json / tokenizer /
#                                              processor_config / chat_template
# merger 会读 `actor/huggingface/` 里的 config 作为 hf_model_config_path，
# 再把重建出的 state_dict 通过 `model.save_pretrained(target_dir)` 写出去。
#
# 用法：
#   # 合并单个 step（默认原地写入 actor/huggingface/）
#   STEPS=100 bash train_verl/utils/ckpt/merge_fsdp_ckpt_to_hf.sh
#
#   # 批量合并
#   STEPS="25 50 75 100 125" bash train_verl/utils/ckpt/merge_fsdp_ckpt_to_hf.sh
#
#   # 合并 CKPT_ROOT 下所有已存在的 step
#   STEPS=all bash train_verl/utils/ckpt/merge_fsdp_ckpt_to_hf.sh
#
#   # 写到独立目录，不覆盖原 huggingface/
#   TARGET_DIR_MODE=new STEPS=100 bash train_verl/utils/ckpt/merge_fsdp_ckpt_to_hf.sh
#     -> 输出到 actor/huggingface_merged/
#
#   # 换一个实验
#   CKPT_ROOT=/path/to/other_exp/ckpt STEPS=all bash train_verl/utils/ckpt/merge_fsdp_ckpt_to_hf.sh
# =============================================================================
set -euo pipefail

# -------- 用户可覆盖的参数 --------
CKPT_ROOT=${CKPT_ROOT:-/path/to/train_verl/outputs/hyocr_1_5_verl/hyocr_1_5_grpo/ckpt}
STEPS=${STEPS:-all}
# inplace: 直接写到 actor/huggingface/ 目录（与训练时 save_contents+=hf_model 行为一致）
# new    : 写到 actor/huggingface_merged/，不覆盖原目录
TARGET_DIR_MODE=${TARGET_DIR_MODE:-inplace}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
# 大模型加载不下时打开：合并阶段用 CPU init 避免 GPU OOM
USE_CPU_INIT=${USE_CPU_INIT:-1}
# 单卡就够，用哪块 GPU（避免 verl 的其他训练进程抢卡）
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# -------- Conda 环境（与训练脚本对齐） --------
CONDA_PATH=${CONDA_PATH:-/path/to/miniconda3}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-verl-hyocr}
set +u
if [ -f "${CONDA_PATH}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_PATH}/bin/activate"
    conda activate "${CONDA_ENV_NAME}"
fi
set -u

export CUDA_VISIBLE_DEVICES
# merger 内部会 init 一个 hf 模型，用 bf16 + cpu init 内存更省
export PYTHONUNBUFFERED=1

# -------- 组装 STEPS 列表 --------
if [ "${STEPS}" = "all" ]; then
    if [ ! -d "${CKPT_ROOT}" ]; then
        echo "ERROR: CKPT_ROOT does not exist: ${CKPT_ROOT}" >&2
        exit 1
    fi
    steps_list=""
    for d in "${CKPT_ROOT}"/global_step_*; do
        [ -d "${d}" ] || continue
        name=$(basename "${d}")
        case "${name}" in
            global_step_*[!0-9]*) continue ;;
            global_step_*)        steps_list+="${name#global_step_} " ;;
        esac
    done
    steps_list=$(echo "${steps_list}" | tr ' ' '\n' | sort -n | tr '\n' ' ')
    if [ -z "${steps_list// /}" ]; then
        echo "ERROR: no global_step_* found under ${CKPT_ROOT}" >&2
        exit 1
    fi
else
    steps_list="${STEPS}"
fi

echo "==================== merge_fsdp_ckpt_to_hf ===================="
echo "CKPT_ROOT       = ${CKPT_ROOT}"
echo "STEPS           = ${steps_list}"
echo "TARGET_DIR_MODE = ${TARGET_DIR_MODE}"
echo "USE_CPU_INIT    = ${USE_CPU_INIT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "==============================================================="

extra_flags=()
if [ "${TRUST_REMOTE_CODE}" = "1" ] || [ "${TRUST_REMOTE_CODE}" = "True" ] || [ "${TRUST_REMOTE_CODE}" = "true" ]; then
    extra_flags+=("--trust-remote-code")
fi
if [ "${USE_CPU_INIT}" = "1" ] || [ "${USE_CPU_INIT}" = "True" ] || [ "${USE_CPU_INIT}" = "true" ]; then
    extra_flags+=("--use_cpu_initialization")
fi

for step in ${steps_list}; do
    actor_dir="${CKPT_ROOT}/global_step_${step}/actor"
    if [ ! -d "${actor_dir}" ]; then
        echo "[step ${step}] SKIP: ${actor_dir} not found"
        continue
    fi

    # 校验 sharded 权重与 huggingface config 都齐全
    shard_count=0
    for shard in "${actor_dir}"/model_world_size_*_rank_*.pt; do
        [ -f "${shard}" ] && shard_count=$((shard_count + 1))
    done
    if [ "${shard_count}" -eq 0 ]; then
        echo "[step ${step}] SKIP: no model_world_size_*_rank_*.pt shards in ${actor_dir}"
        continue
    fi
    if [ ! -f "${actor_dir}/huggingface/config.json" ]; then
        echo "[step ${step}] SKIP: missing ${actor_dir}/huggingface/config.json (hf_model_config_path)"
        continue
    fi

    if [ "${TARGET_DIR_MODE}" = "inplace" ]; then
        target_dir="${actor_dir}/huggingface"
    else
        target_dir="${actor_dir}/huggingface_merged"
    fi

    # 已经合并过就跳过（存在 model.safetensors 或 model.safetensors.index.json）
    if [ -f "${target_dir}/model.safetensors" ] || [ -f "${target_dir}/model.safetensors.index.json" ]; then
        echo "[step ${step}] SKIP: hf weights already present at ${target_dir}"
        continue
    fi

    echo "[step ${step}] merging ${shard_count} shards -> ${target_dir}"
    mkdir -p "${target_dir}"

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${actor_dir}" \
        --target_dir "${target_dir}" \
        "${extra_flags[@]}"

    echo "[step ${step}] done"
    find "${target_dir}" -maxdepth 1 -mindepth 1 -printf "%f\t%s bytes\n" 2>/dev/null | head -20
done

echo "all steps merged."
