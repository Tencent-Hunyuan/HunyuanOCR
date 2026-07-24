#!/bin/bash
# shellcheck disable=SC1091
#
# Start a Ray cluster (head + optional slave nodes) for verl training.
#
# Usage:
#   bash start_ray.sh                      # multi-node, slaves from NODE_IP_LIST
#   bash start_ray.sh <HOST_NUM> <INDEX>   # multi-node, slaves from $HOME/hosts/pssh.hosts_${HOST_NUM}node_${INDEX}
#   bash start_ray.sh local                # single-node
#   bash start_ray.sh [local|NUM INDEX] [EXTRA_ENV...]
#
# Env vars expected (usually exported by the training platform; set manually if
# running standalone):
#   LOCAL_IP        this node's IP
#   NODE_IP_LIST    space-separated list of all node IPs (default mode only)

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
MAX_RAY_WAIT_TIME=60
HOSTS_DIR="$HOME/hosts"
mkdir -p "$HOSTS_DIR"

CONDA_PATH=${CONDA_PATH:-/path/to/miniconda3}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-verl-hyocr}

ARG1="$1"
ARG2="$2"

source "${SCRIPT_DIR}/network_envs.sh"

# Activate conda env on the head node.
CONDA_INIT_CMD=""
if [ -f "${CONDA_PATH}/bin/activate" ]; then
    source "${CONDA_PATH}/bin/activate"
    if conda activate "${CONDA_ENV_NAME}"; then
        echo "Conda env '${CONDA_ENV_NAME}' activated on head node."
    else
        echo "Error: failed to activate conda env '${CONDA_ENV_NAME}'"
        exit 1
    fi
    CONDA_INIT_CMD="source ${CONDA_PATH}/bin/activate && conda activate ${CONDA_ENV_NAME};"
else
    echo "Error: conda not found at ${CONDA_PATH}"
    exit 1
fi

# Build a slave-only hosts file (excluding the head) for pssh.
if [ "$ARG1" = "local" ]; then
    EXTRA_ENV="${*:2}"
    USE_PSSH=false
    echo "Single-node mode; only start head."

elif [ -n "$ARG1" ] && [ -n "$ARG2" ] && [[ "$ARG1" =~ ^[0-9]+$ ]] && [[ "$ARG2" =~ ^[0-9]+$ ]]; then
    FULL_HOSTS_FILE="$HOSTS_DIR/pssh.hosts_${ARG1}node_${ARG2}"
    PSSH_SLAVE_HOSTS_FILE="$HOSTS_DIR/pssh.hosts_${ARG1}node_${ARG2}_slave"
    echo "Using hosts file: ${FULL_HOSTS_FILE}"

    if [ ! -f "${FULL_HOSTS_FILE}" ]; then
        echo "Error: hosts file '${FULL_HOSTS_FILE}' not found"
        exit 1
    fi

    grep -v "^${LOCAL_IP}$" "${FULL_HOSTS_FILE}" >"${PSSH_SLAVE_HOSTS_FILE}"
    echo "Wrote slave hosts file: ${PSSH_SLAVE_HOSTS_FILE}"
    cat "${PSSH_SLAVE_HOSTS_FILE}"

    EXTRA_ENV="${*:3}"
    USE_PSSH=true

else
    # Derive slave hosts from NODE_IP_LIST.
    PSSH_SLAVE_HOSTS_FILE=$(mktemp "$HOSTS_DIR/pssh.hosts_slave_XXXXXX")
    echo "$NODE_IP_LIST" |
        sed 's/:[^,]*//g' |
        sed 's/,/\n/g' |
        sort -u |
        grep -v "^${LOCAL_IP}$" \
            >"${PSSH_SLAVE_HOSTS_FILE}"
    echo "Derived slave hosts file: ${PSSH_SLAVE_HOSTS_FILE}"
    cat "${PSSH_SLAVE_HOSTS_FILE}"

    EXTRA_ENV="${*:1}"
    USE_PSSH=true
fi

# Head node: pin the Ray version so a non-official dev build in the env cannot
# trigger GCS startup failures.
RAY_VERSION="${RAY_VERSION:-2.46.0}"
ENV_PIP="${CONDA_PATH}/envs/${CONDA_ENV_NAME}/bin/pip"
ENV_PYTHON="${CONDA_PATH}/envs/${CONDA_ENV_NAME}/bin/python"
CURRENT_RAY_VERSION=$("${ENV_PYTHON}" -c "import ray; print(ray.__version__)" 2>/dev/null | head -1)
if [ "${CURRENT_RAY_VERSION}" != "${RAY_VERSION}" ] || [[ "${CURRENT_RAY_VERSION}" == *"+"* ]]; then
    echo "Reinstalling ray==${RAY_VERSION} into ${CONDA_ENV_NAME} (current: ${CURRENT_RAY_VERSION})"
    "${ENV_PIP}" uninstall -y ray ray-cpp 2>/dev/null || true
    "${ENV_PIP}" install --no-cache-dir "ray[default]==${RAY_VERSION}" --root-user-action ignore
fi
if [ -n "$EXTRA_ENV" ]; then
    echo "set ${EXTRA_ENV} for ray head node process"
    eval "${EXTRA_ENV}" ray start --head --node-ip-address "${LOCAL_IP}" --num-gpus 8 --disable-usage-stats
else
    ray start --head --node-ip-address "${LOCAL_IP}" --num-gpus 8 --disable-usage-stats
fi

if [[ "${CI_TEST,,}" == "true" ]]; then
    CI_TEST=true
    DETERMINISTIC_MODE=true
else
    CI_TEST=false
fi

# Slave nodes.
if [ "$USE_PSSH" = true ]; then
    if [ -n "${EXTRA_ENV}" ]; then
        echo "set ${EXTRA_ENV} for ray slave node process"
        pssh -i -t 0 -h "${PSSH_SLAVE_HOSTS_FILE}" \
            "export CI_TEST=${CI_TEST} DETERMINISTIC_MODE=${DETERMINISTIC_MODE}; \
            ${CONDA_INIT_CMD} \
            source ${SCRIPT_DIR}/network_envs.sh; \
            eval \"$EXTRA_ENV\" ray start --address ${LOCAL_IP}:6379 --num-gpus 8"
    else
        pssh -i -t 0 -h "${PSSH_SLAVE_HOSTS_FILE}" \
            "export CI_TEST=${CI_TEST} DETERMINISTIC_MODE=${DETERMINISTIC_MODE}; \
            ${CONDA_INIT_CMD} \
            source ${SCRIPT_DIR}/network_envs.sh; \
            ray start --address ${LOCAL_IP}:6379 --num-gpus 8"
    fi
else
    echo "Single-node mode; skipping slave startup."
fi

# Wait for Ray to be ready.
start_time=$(date +%s)
while true; do
    if ray status >/dev/null 2>&1; then
        echo "Ray has been successfully started"
        break
    fi

    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [ $elapsed_time -ge $MAX_RAY_WAIT_TIME ]; then
        echo "Ray startup timed out; check that Ray is installed correctly"
        exit 1
    fi

    echo "Waiting for Ray to start..."
    sleep 3
done
echo "Ray has been successfully started; continuing"
