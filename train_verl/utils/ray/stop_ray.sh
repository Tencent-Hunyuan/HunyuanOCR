#!/bin/bash
# shellcheck disable=SC1091

# Usage:
#   bash stop_ray.sh                      # 使用 $HOME/hosts/pssh.hosts
#   bash stop_ray.sh <HOST_NUM> <INDEX>   # 使用 hosts/pssh.hosts_${HOST_NUM}node_${INDEX}
#   bash stop_ray.sh local                # 使用本机 LOCAL_IP

set -e

# Optional: source a platform env file if present (skipped on standalone hosts).
_PLATFORM_ENV_FILE=${PLATFORM_ENV_FILE:-}
[ -n "${_PLATFORM_ENV_FILE}" ] && [ -f "${_PLATFORM_ENV_FILE}" ] && source "${_PLATFORM_ENV_FILE}"
HOSTS_DIR="$HOME/hosts"
mkdir -p "$HOSTS_DIR"

ARG1="$1"
ARG2="$2"

if [ "$ARG1" = "local" ]; then
    # 单机模式：写入本机 IP
    PSSH_HOSTS_FILE="$HOSTS_DIR/pssh.hosts_local"
    echo "$LOCAL_IP" >"${PSSH_HOSTS_FILE}"
    echo "使用本机 hosts 文件: ${PSSH_HOSTS_FILE} (${LOCAL_IP})"
elif [ -n "$ARG1" ] && [ -n "$ARG2" ]; then
    # 指定节点数和索引
    PSSH_HOSTS_FILE="$HOSTS_DIR/pssh.hosts_${ARG1}node_${ARG2}"
    echo "使用 hosts 文件: ${PSSH_HOSTS_FILE}"
else
    # 默认使用 pssh.hosts
    PSSH_HOSTS_FILE="$HOSTS_DIR/pssh.hosts"
    echo "使用默认 hosts 文件: ${PSSH_HOSTS_FILE}"
fi

# perform silent remote stop/kill (suppress long output)
echo "停止 Ray 服务..."
pssh -t 0 -h "${PSSH_HOSTS_FILE}" "ray stop --force" || true
echo "停止 Triton 服务..."
pssh -P -t 0 -h "${PSSH_HOSTS_FILE}" "killall tritonserver" || true
# echo "停止 Python 进程..."
# pssh -P -t 0 -h "${PSSH_HOSTS_FILE}" "pkill -9 python" || true

echo "停止命令已发送 (目标 hosts: ${PSSH_HOSTS_FILE})"
