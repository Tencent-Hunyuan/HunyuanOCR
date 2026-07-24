#!/bin/bash
# shellcheck disable=SC1091,SC2155
# ============================================================================
# HunyuanOCR-1.5 GRPO training on verl (FSDP2 actor + vLLM async rollout).
# ============================================================================
set -xeuo pipefail

# ---------------- Paths (override via env) ----------------
MODEL_PATH=${MODEL_PATH:-/path/to/HunyuanOCR/model}
TRAIN_FILES=${TRAIN_FILES:-/path/to/train.parquet}
VAL_FILES=${VAL_FILES:-/path/to/val.parquet}
# Output root: {ckpt, tensorboard, file logger, stdout log, rl_logging_board dumps}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs}

NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
project_name=${PROJECT_NAME:-hyocr_1_5_verl}
experiment_name=${EXPERIMENT_NAME:-hyocr_1_5_grpo}

# ---- Conda env ----
export CONDA_PATH=${CONDA_PATH:-/path/to/miniconda3}
export CONDA_ENV_NAME=${CONDA_ENV_NAME:-verl-hyocr}

# ---------------- Batching ----------------
train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
rollout_n=${ROLLOUT_N:-16}
save_freq=${SAVE_FREQ:-25}
val_freq=${VAL_FREQ:-100000}

# ---------------- Checkpoint contents ----------------
# save_hf_model=True  -> also dump a merged HuggingFace-loadable checkpoint at
#   ${ckpt}/global_step_N/actor/huggingface/ every save_freq (~2× disk per step;
#   set to False + use utils/ckpt/merge_fsdp_ckpt_to_hf.sh for offline merging).
save_hf_model=${SAVE_HF_MODEL:-True}
if [ "${save_hf_model}" = "True" ] || [ "${save_hf_model}" = "true" ] || [ "${save_hf_model}" = "1" ]; then
    checkpoint_save_contents="[model,optimizer,extra,hf_model]"
else
    checkpoint_save_contents="[model,optimizer,extra]"
fi

# ---------------- Sequence lengths ----------------
max_prompt_length=${MAX_PROMPT_LENGTH:-8192}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
sequence_length=${SEQUENCE_LENGTH:-16384}
log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}
ref_log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
use_remove_padding=${USE_REMOVE_PADDING:-False}
reward_num_workers=${REWARD_NUM_WORKERS:-8}
enforce_eager=${enforce_eager:-False}

# ---------------- Dynamic batch size ----------------
# dynamic bsz packs each rank's micro-batch by token count. When rank
# micro-batch counts diverge, faster ranks stall on the next FSDP2 all-gather
# until the NCCL watchdog fires. Default on; force mb=1 to keep ranks aligned.
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
actor_micro_bsz_per_gpu=${ACTOR_MICRO_BSZ_PER_GPU:-1}
log_prob_micro_bsz_per_gpu=${LOG_PROB_MICRO_BSZ_PER_GPU:-1}

# ---------------- Optim ----------------
actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
total_training_steps=${TOTAL_TRAINING_STEPS:-100000}

# ---------------- Rollout (vLLM) ----------------
rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}

# ---------------- Fast vs slow HF processor ----------------
# transformers>=5 defaults to use_fast=True for multimodal processors, which
# switches the image processor from PIL/numpy to a Rust/Torch backend.
use_fast_processor=${USE_FAST_PROCESSOR:-False}

# ---------------- Vision special-token guards ----------------
# Two switches keep the vLLM rollout distribution aligned with the actor:
#   * bad_words: 6 vision control tokens are masked to -inf in vLLM sampling.
#   * add_vision_logit_bias: an additional -1e9 bias is added before softmax so
#     they are strictly zero-probability under fp32 sampling.
# Without these, the model can sample e.g. <｜hy_image▁pad｜>, which routes
# the actor forward through the image position / attention branch and blows up
# training/rollout_probs_diff_mean.
add_vision_logit_bias=${ADD_VISION_LOGIT_BIAS:-True}
bad_words=${BAD_WORDS:-'["<｜hy_image▁pad｜>","<｜hy_image▁start｜>","<｜hy_image▁end｜>","<｜hy_video▁pad｜>","<｜hy_video▁start｜>","<｜hy_video▁end｜>"]'}

# ---------------- IcePop (rollout correction) ----------------
# Mapped from the internal `policy_loss.loss_mode: icepop, imp_ratio_cap_range:
# [alpha, beta]`. verl's `algorithm.rollout_correction.decoupled_token_icepop`
# implements the same clamp: zero the IS weights of tokens outside [alpha, beta]
# while keeping response_mask intact.
icepop_lower=${ICEPOP_LOWER:-0.2}
icepop_upper=${ICEPOP_UPPER:-5.0}

# ---------------- Output layout ----------------
run_dir="${OUTPUT_DIR}/${project_name}/${experiment_name}"
default_local_dir=${DEFAULT_LOCAL_DIR:-${run_dir}/ckpt}
tb_dir=${TENSORBOARD_DIR:-${run_dir}/tb}
log_dir="${run_dir}/logs"
# RLLoggingBoard per-step rollout dumps. View with:
#   streamlit run rl_logging_board/app.py -- --logdir ${rlb_dir}
rlb_dir=${RLB_DIR:-${run_dir}/rl_logging_board}
rlb_enable=${RLB_ENABLE:-True}
rlb_max_response_tokens=${RLB_MAX_RESPONSE_TOKENS:-4096}

# ---------------- Dataset filter ----------------
max_workers=${MAX_WORKERS:-64}
# Cache the "filtered out overlong prompts" dataset on disk so re-runs skip the
# expensive filter. Cache key = data_files + max_prompt_length + max_samples +
# tokenizer + seed + apply_chat_template_kwargs (md5).
dataset_cache_enable=${DATASET_CACHE_ENABLE:-True}
dataset_cache_dir=${DATASET_CACHE_DIR:-./dataset_cache}
dataset_cache_force_rebuild=${DATASET_CACHE_FORCE_REBUILD:-False}

mkdir -p "${default_local_dir}" "${tb_dir}" "${log_dir}" "${rlb_dir}"
timestamp="$(date +%Y%m%d-%H%M%S)"
stdout_log="${log_dir}/${timestamp}.log"

# verl reads TB / file-logger dirs from env
export TENSORBOARD_DIR="${tb_dir}"
export VERL_FILE_LOGGER_PATH="${run_dir}/metrics.jsonl"

REWARD_FN_PATH=${REWARD_FN_PATH:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/reward_ocr.py"}
########################### end user-adjustable ###########################

# Make the `reward` package importable from reward_ocr.py
export PYTHONPATH="$(dirname "${REWARD_FN_PATH}"):${PYTHONPATH:-}"
export RM_SYSTEM_JUDGE_MODEL_NAME=${RM_SYSTEM_JUDGE_MODEL_NAME:-Qwen/Qwen3-30B-A3B}

# Ray log settings (verl example defaults). RAY_DEDUP_LOGS=1 disables Ray's
# stdout de-dup so remote worker prints show up on the driver in real time.
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# ---- Ray cluster: start-and-attach ----
# Bring up an external Ray head via utils/ray/start_ray.sh, then let this
# driver attach through RAY_ADDRESS=auto. Decoupling the cluster from the
# training process keeps the dashboard / slave nodes alive across driver
# restarts.
#
#   START_RAY=1          start_ray.sh first; default 1
#   RAY_START_MODE=local passed as arg1 to start_ray.sh (local / multi-node)
#   RAY_INDEX            passed as arg2 for multi-node runs
#   STOP_RAY_AFTER=0     stop_ray.sh after training; default 0 (keep for logs)
#   NETWORK_ENV_SCRIPT   NCCL/IB env script; default utils/ray/network_envs.sh
START_RAY=${START_RAY:-1}
RAY_START_MODE=""
STOP_RAY_AFTER=${STOP_RAY_AFTER:-0}

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NETWORK_ENV_SCRIPT=${NETWORK_ENV_SCRIPT:-"${SCRIPT_DIR}/utils/ray/network_envs.sh"}
export START_RAY_SCRIPT=${START_RAY_SCRIPT:-"${SCRIPT_DIR}/utils/ray/start_ray.sh"}
export STOP_RAY_SCRIPT=${STOP_RAY_SCRIPT:-"${SCRIPT_DIR}/utils/ray/stop_ray.sh"}

# Cluster env scripts may reference variables that are only defined on the
# training platform; disable nounset while sourcing them.
set +u
# Optional: source a platform env file if present (skipped on standalone hosts).
_PLATFORM_ENV_FILE=${PLATFORM_ENV_FILE:-}
[ -n "${_PLATFORM_ENV_FILE}" ] && [ -f "${_PLATFORM_ENV_FILE}" ] && source "${_PLATFORM_ENV_FILE}"
if [ -f "${NETWORK_ENV_SCRIPT}" ]; then
    # shellcheck disable=SC1090
    source "${NETWORK_ENV_SCRIPT}"
fi
set -u

set +u
if [ -f "${CONDA_PATH}/bin/activate" ]; then
    source "${CONDA_PATH}/bin/activate"
    if conda activate "${CONDA_ENV_NAME}"; then
        echo "Conda env '${CONDA_ENV_NAME}' activated (python: $(command -v python))"
    else
        echo "Error: failed to activate conda env '${CONDA_ENV_NAME}'"
        exit 1
    fi
else
    echo "Error: conda not found at ${CONDA_PATH}"
    exit 1
fi
set -u

if [ "${START_RAY}" = "1" ]; then
    echo "---- (re)starting ray cluster via ${START_RAY_SCRIPT} ${RAY_START_MODE} ${RAY_INDEX:-} ----"
    if ray status >/dev/null 2>&1; then
        echo "existing ray cluster detected, stopping first..."
        bash "${STOP_RAY_SCRIPT}" "${RAY_START_MODE}" "${RAY_INDEX:-}" || true
        sleep 3
    fi
    # shellcheck disable=SC2086
    bash "${START_RAY_SCRIPT}" ${RAY_START_MODE} ${RAY_INDEX:-}
fi

# Attach the driver to the external head instead of starting a local cluster.
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
echo "RAY_ADDRESS=${RAY_ADDRESS}"
ray status || {
    echo "Error: ray cluster not reachable via RAY_ADDRESS=${RAY_ADDRESS}"
    exit 1
}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# vLLM / torchinductor / flashinfer caches on shared fast storage.
_CACHE_BASE=${_CACHE_BASE:-./startup_cache}
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${_CACHE_BASE}/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${_CACHE_BASE}/torchinductor}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${_CACHE_BASE}/flashinfer}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_WORKSPACE_DIR}"

echo "==================== HunyuanOCR-1.5 GRPO ===================="
echo "run_dir      = ${run_dir}"
echo "  ckpt       -> ${default_local_dir}"
echo "  tensorboard-> ${tb_dir}"
echo "  file logger-> ${VERL_FILE_LOGGER_PATH}"
echo "  stdout log -> ${stdout_log}"
echo "  rlb dumps  -> ${rlb_dir}  (enable=${rlb_enable})"
echo "  ds cache   -> ${dataset_cache_dir}  (enable=${dataset_cache_enable}, force_rebuild=${dataset_cache_force_rebuild})"
echo "  filter proc-> num_proc=${max_workers}"
echo "  vllm cache -> ${VLLM_CACHE_ROOT}"
echo "  dyn_bsz    -> ${use_dynamic_bsz}  (actor_mb/gpu=${actor_micro_bsz_per_gpu}, logp_mb/gpu=${log_prob_micro_bsz_per_gpu})"
echo "  save_hf    -> ${save_hf_model}  (checkpoint.save_contents=${checkpoint_save_contents})"
echo "  vision guard-> add_vision_logit_bias=${add_vision_logit_bias}"
echo "               bad_words=${bad_words}"
echo "============================================================="

# tee stdout+stderr to log while still printing to terminal
exec > >(tee -a "${stdout_log}") 2>&1

set +e
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold="${icepop_lower}_${icepop_upper}" \
    algorithm.rollout_correction.rollout_is_batch_normalize=False \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.rollout_rs_threshold=null \
    algorithm.rollout_correction.bypass_mode=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.image_key=images \
    data.train_batch_size="${train_batch_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers="${max_workers}" \
    +data.enable_dataset_cache="${dataset_cache_enable}" \
    +data.dataset_cache_dir="${dataset_cache_dir}" \
    +data.dataset_cache_force_rebuild="${dataset_cache_force_rebuild}" \
    data.truncation='error' \
    data.shuffle=True \
    data.trust_remote_code=True \
    +data.use_fast_processor="${use_fast_processor}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_fast_processor="${use_fast_processor}" \
    actor_rollout_ref.model.use_remove_padding="${use_remove_padding}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${actor_lr}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ppo_max_token_len_per_gpu}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=1.5 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents="${checkpoint_save_contents}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${ref_log_prob_max_token_len_per_gpu}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager="${enforce_eager}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${rollout_tp}" \
    actor_rollout_ref.rollout.prompt_length="${max_prompt_length}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_mem_util}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${sequence_length}" \
    actor_rollout_ref.rollout.max_model_len="${sequence_length}" \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.n="${rollout_n}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${log_prob_max_token_len_per_gpu}" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    +actor_rollout_ref.rollout.add_vision_logit_bias="${add_vision_logit_bias}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=5 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True \
    +actor_rollout_ref.rollout.bad_words="${bad_words}" \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.num_workers="${reward_num_workers}" \
    trainer.balance_batch=True \
    trainer.logger='["console","tensorboard","file"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq="${save_freq}" \
    trainer.test_freq="${val_freq}" \
    trainer.val_before_train=False \
    trainer.total_training_steps="${total_training_steps}" \
    trainer.total_epochs=100 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${default_local_dir}" \
    +trainer.rl_logging_board.enable="${rlb_enable}" \
    +trainer.rl_logging_board.dump_dir="${rlb_dir}" \
    +trainer.rl_logging_board.max_response_tokens="${rlb_max_response_tokens}" \
    +trainer.rl_logging_board.include_response_tokens=True \
    +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="${tb_dir}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH="${run_dir}/metrics.jsonl" \
    "$@"
TRAIN_EXIT=$?
set -e

if [ "${STOP_RAY_AFTER}" = "1" ]; then
    echo "---- stopping ray cluster via ${STOP_RAY_SCRIPT} ${RAY_START_MODE} ${RAY_INDEX:-} ----"
    bash "${STOP_RAY_SCRIPT}" "${RAY_START_MODE}" "${RAY_INDEX:-}" || true
fi

exit "${TRAIN_EXIT}"
