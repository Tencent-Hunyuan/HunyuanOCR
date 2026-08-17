#!/bin/bash
# ============================================================================
# Environment for SINGLE-NODE / SINGLE-GPU training.
#
# Use this instead of env_common.sh when there is no InfiniBand fabric.
# env_common.sh pins NCCL to a `bond1` interface and eight mlx5 HCAs; on a
# single box those devices do not exist and NCCL either warns loudly or hangs
# during rendezvous.
# ============================================================================

# No IB, no multi-NIC: keep NCCL on loopback and let it fall back to shared mem.
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=1

# Tokenizers forks inside the dataloader workers; silence the parallelism warning.
export TOKENIZERS_PARALLELISM=false

# Less fragmentation when activations swing between packs of different length.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
