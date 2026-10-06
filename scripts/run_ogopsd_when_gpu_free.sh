#!/usr/bin/env bash
# Wait for an exclusive idle GPU (no other processes), then launch OG-OPSD
# λ=1 100-step training. Never attaches to a GPU that already has compute.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
WAIT_LOG="${LOG_DIR}/ogopsd_wait_${STAMP}.log"
TRAIN_LOG="${LOG_DIR}/ogopsd_train_${STAMP}.log"
META="${LOG_DIR}/ogopsd_train_${STAMP}.meta"
GPU_FILE="${LOG_DIR}/ogopsd_gpu_${STAMP}.txt"

export WANDB_MODE="${WANDB_MODE:-offline}"
export OUTCOME_GATE_LAMBDA="${OUTCOME_GATE_LAMBDA:-1}"
export RUN_CONFIG="${RUN_CONFIG:-ogopsd_lambda1_100}"
export MAX_STEPS="${MAX_STEPS:-100}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-13949}"

echo "waiting for exclusive idle GPU..." | tee "${WAIT_LOG}"
GPU_IDX="$(MAX_USED_MIB=400 POLL=20 bash "${REPO_DIR}/scripts/wait_exclusive_gpu.sh" | tee -a "${WAIT_LOG}" | tail -n 1)"
if [[ -z "${GPU_IDX}" || ! "${GPU_IDX}" =~ ^[0-9]+$ ]]; then
    echo "failed to obtain exclusive GPU: '${GPU_IDX}'" | tee -a "${WAIT_LOG}"
    exit 2
fi

# Re-check immediately before launch (race with other jobs).
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_IDX}" | awk '{print int($1)}')"
NPROC="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${GPU_IDX}" | sed '/^\s*$/d' | wc -l)"
if [[ "${NPROC}" -gt 0 || "${USED}" -gt 400 ]]; then
    echo "GPU ${GPU_IDX} was claimed before launch (used=${USED}MiB nproc=${NPROC})" | tee -a "${WAIT_LOG}"
    exit 3
fi

echo "${GPU_IDX}" > "${GPU_FILE}"
{
    echo "gpu=${GPU_IDX}"
    echo "run_config=${RUN_CONFIG}"
    echo "lambda=${OUTCOME_GATE_LAMBDA}"
    echo "train_log=${TRAIN_LOG}"
    echo "started=$(date -Is)"
} > "${META}"

echo "launching OG-OPSD on exclusive GPU ${GPU_IDX}" | tee -a "${WAIT_LOG}"
cd "${REPO_DIR}"
set +e
CUDA_DEVICES="${GPU_IDX}" NUM_PROCESSES=1 \
    OUTCOME_GATE_LAMBDA="${OUTCOME_GATE_LAMBDA}" \
    RUN_CONFIG="${RUN_CONFIG}" \
    MAX_STEPS="${MAX_STEPS}" \
    MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
    WANDB_MODE="${WANDB_MODE}" \
    bash scripts/run_opsd_1b_local.sh >"${TRAIN_LOG}" 2>&1
RC=$?
set -e
echo "exit=${RC}" >> "${META}"
echo "finished=$(date -Is)" >> "${META}"
exit "${RC}"
