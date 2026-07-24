#!/bin/bash
# NCCL / IB env template for multi-node Ray + verl training.
#
# The exact HCA / socket interface names depend on your cluster. Adjust the
# NCCL_IB_HCA / *_SOCKET_IFNAME values below to match `ibdev2netdev` and
# `ip -o link show` on your nodes.
#
# Fallback tiers (top to bottom, if the layer above breaks on your driver /
# MOFED combination):
#   A. IB + GPUDirect RDMA on           (fastest; requires modern rdma-core +
#      compatible NCCL/driver).
#   B. IB on but GDR / DMA-BUF off      (safe default; host bounce buffers).
#   C. Plain TCP over Ethernet          (enabled via `DISABLE_IB=1`).

export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export HYDRA_FULL_ERROR=1
export RAY_prestart_worker_first_driver=false
export RAY_memory_monitor_refresh_ms=0

if [[ "${CI_TEST,,}" == "true" ]] || [[ "${DETERMINISTIC_MODE,,}" == "true" ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export FLASH_ATTENTION_DETERMINISTIC=1
    export NCCL_ALGO=Ring
    export CUDA_LAUNCH_BLOCKING=1
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
    export TORCHDYNAMO_DISABLE=1
fi

# ---- Tier selection -------------------------------------------------------
# Set the socket interface name to whichever NIC handles rendezvous on your
# cluster (e.g. eth0 / bond0 / bond1). NCCL_IB_HCA should list your IB devices.
SOCK_IFNAME=${SOCK_IFNAME:-bond1}
IB_HCA=${IB_HCA:-mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8}

if [[ "${DISABLE_IB:-0}" == "1" ]]; then
    # Tier C: plain TCP.
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_SOCKET_IFNAME=${SOCK_IFNAME}
    export GLOO_SOCKET_IFNAME=${SOCK_IFNAME}
else
    # Tier B (default): IB on, GDR / DMA-BUF off. Safe on most driver combos.
    # Set NCCL_NET_GDR_LEVEL=2 (or 3) to opt into Tier A once you have verified
    # GPUDirect RDMA works end-to-end on your cluster.
    export NCCL_IB_DISABLE=0
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_HCA=${IB_HCA}
    export NCCL_IB_GID_INDEX=3
    export NCCL_NET_GDR_LEVEL=0
    export NCCL_DMABUF_ENABLE=0
    export NCCL_IB_GDR_FLUSH_DISABLE=1
    export NCCL_SOCKET_IFNAME=${SOCK_IFNAME}
    export GLOO_SOCKET_IFNAME=${SOCK_IFNAME}
fi

# Uncomment for verbose NCCL logs while debugging transport / rendezvous.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET
