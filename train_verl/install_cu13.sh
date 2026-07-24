#!/bin/bash
#
# CUDA 13 下环境一键安装脚本
#
# 前置条件（宿主机需已经具备）：
#   - NVIDIA Driver >= 535.161.08
#   - CUDA >= 13.0 (推荐 13.3)
#   - Python 3.12.11
#   - 系统级 cuDNN、NCCL、nsight、ffmpeg 等由宿主机/镜像提供
#
# 版本组合参考 verl/docker/Dockerfile.stable.vllm，与 CUDA 13 对齐。
# 与 install_vllm_sglang_mcore.sh 保持同样的分步结构，方便对照。
#
# Usage:
#   USE_MEGATRON=0 USE_SGLANG=0 bash install_cu13.sh
#

set -euxo pipefail

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

# ============================================================
# 版本对齐 Dockerfile.stable.vllm (CUDA 13)
# ============================================================
TORCH_VERSION=${TORCH_VERSION:-2.11.0}
TORCH_VISION_VERSION=${TORCH_VISION_VERSION:-0.26.0}
TORCH_AUDIO_VERSION=${TORCH_AUDIO_VERSION:-2.11.0}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}

TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-5.13.0}
VLLM_VERSION=${VLLM_VERSION:-0.25.1}
SGLANG_VERSION=${SGLANG_VERSION:-0.5.2}
TRL_VERSION=${TRL_VERSION:-0.27.0}
FLASH_ATTENTION_VERSION=${FLASH_ATTENTION_VERSION:-2.8.3}
TRANSFORMER_ENGINE_VERSION=${TRANSFORMER_ENGINE_VERSION:-v2.15}
MCORE_VERSION=${MCORE_VERSION:-core_v0.18.0}
QWEN_VL_UTILS_VERSION=${QWEN_VL_UTILS_VERSION:-0.0.14}
FLASHINFER_VERSION=${FLASHINFER_VERSION:-0.3.1}
MEGATRON_BRIDGE_VERSION=${MEGATRON_BRIDGE_VERSION:-0.5.0}

# PEP 668: Ubuntu 24.04 blocks system-wide pip installs; override if in docker/root env
export PIP_BREAK_SYSTEM_PACKAGES=${PIP_BREAK_SYSTEM_PACKAGES:-1}

pip install -U pip uv wheel pybind11 ninja

echo "0. install PyTorch (cu130)"
uv pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCH_VISION_VERSION}" \
    "torchaudio==${TORCH_AUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

echo "1. install inference frameworks"
if [ "$USE_SGLANG" -eq 1 ]; then
    uv pip install "sglang[all]==${SGLANG_VERSION}" && uv pip install torch-memory-saver
fi
uv pip install "vllm==${VLLM_VERSION}"

echo "2. install basic packages"
uv pip install "transformers[hf_xet]==${TRANSFORMERS_VERSION}" accelerate datasets peft hf-transfer \
    "numpy>=2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc "qwen-vl-utils==${QWEN_VL_UTILS_VERSION}" wandb dill pybind11 liger-kernel mathruler \
    pytest pytest-asyncio py-spy pre-commit ruff tensorboard \
    nvtx matplotlib cachetools

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "uv pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "3. install FlashAttention and FlashInfer"
export MAX_JOBS=${MAX_JOBS:-128}
export FLASH_ATTENTION_FORCE_BUILD=TRUE
uv pip install --no-build-isolation --no-cache-dir "flash-attn==${FLASH_ATTENTION_VERSION}"

uv pip install "flashinfer-python==${FLASHINFER_VERSION}"

if [ "$USE_MEGATRON" -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    uv pip install "onnxscript==0.3.1"
    uv pip install nvidia-mathdx

    NVTE_FRAMEWORK=pytorch MAX_JOBS=${MAX_JOBS} NVTE_BUILD_THREADS_PER_JOB=4 \
        uv pip install --no-deps --no-build-isolation \
        "git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_VERSION}"

    echo "4.1 build & install NVIDIA/apex from source (cpp_ext + cuda_ext)"
    echo "Notice that apex source build can take very long time, please be patient"
    # Use pip (not uv pip) here so that --config-settings "--build-option=..." is
    # forwarded correctly to setup.py, matching Dockerfile.stable.vllm.
    MAX_JOBS=${MAX_JOBS} pip install -v --disable-pip-version-check --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" \
        "git+https://github.com/NVIDIA/apex.git"

    # mbridge / megatron-bridge / Megatron-LM
    uv pip install -U "git+https://github.com/ISEEKYAN/mbridge.git@main"
    uv pip install --no-deps "megatron-bridge==${MEGATRON_BRIDGE_VERSION}"
    uv pip install --no-deps "git+https://github.com/NVIDIA/Megatron-LM.git@${MCORE_VERSION}"
fi

echo "5. May need to fix opencv"
uv pip install opencv-python
uv pip install opencv-fixer &&
    python -c "from opencv_fixer import AutoFix; AutoFix()"

if [ "$USE_MEGATRON" -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    # 对齐 cu13
    uv pip install "nvidia-cudnn-cu13"
fi

echo "7. Install trl (no deps to avoid overriding transformers)"
uv pip install --no-deps "trl==${TRL_VERSION}"

echo "8. Install torchcodec (cu130)"
uv pip install torchcodec --index-url="${TORCH_INDEX_URL}" || echo "torchcodec install failed, skip (need ffmpeg on host)"

uv pip install flashinfer-python==0.6.13 orjson --no-deps
uv pip install wandb immutabledict puremagic \
            python-Levenshtein Levenshtein pycocotools \
            harvesttext table_recognition_metric bs4 \
            pyyaml tqdm pillow requests lmdb json_repair \
            openpyxl sseclient retrying jsonlines \
            httpx openai scipy

echo "9. Override NCCL to >= 2.29.7 for ncclCommSuspend / ncclCommResume (cu13)"
# Ref: https://github.com/verl-project/verl/issues/6266
uv pip install --no-deps --upgrade "nvidia-nccl-cu13>=2.29.7,<3.0"

echo "10. TransferQueue"
uv pip install --no-deps TransferQueue==0.1.8

echo "Successfully installed all packages (CUDA 13)"
