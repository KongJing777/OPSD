#!/usr/bin/env bash
# Paper-protocol AIME24 canary on an exclusive idle GPU. Do not share cards.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${CKPT:-${REPO_DIR}/runs/ogopsd_lambda1_100/checkpoint-100}"
BASE="${BASE:-${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}}"
OUT="${OUT:-${REPO_DIR}/eval_results/aime24_ogopsd_lambda1.json}"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}" "$(dirname "${OUT}")"
STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_LOG="${LOG_DIR}/aime24_ogopsd_${STAMP}.log"

if [[ ! -f "${CKPT}/adapter_model.safetensors" ]]; then
    echo "missing adapter at ${CKPT}" >&2
    exit 2
fi

GPU_IDX="$(MAX_USED_MIB=400 POLL=20 bash "${REPO_DIR}/scripts/wait_exclusive_gpu.sh")"
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_IDX}" | awk '{print int($1)}')"
NPROC="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${GPU_IDX}" | sed '/^\s*$/d' | wc -l)"
if [[ "${NPROC}" -gt 0 || "${USED}" -gt 400 ]]; then
    echo "GPU ${GPU_IDX} claimed before eval" >&2
    exit 3
fi

echo "eval AIME24 on exclusive GPU ${GPU_IDX} ckpt=${CKPT}" | tee "${EVAL_LOG}"
cd "${REPO_DIR}/eval"
set +e
CUDA_VISIBLE_DEVICES="${GPU_IDX}" \
python evaluate_math.py \
    --base_model "${BASE}" \
    --checkpoint_dir "${CKPT}" \
    --dataset aime24 \
    --val_n 12 \
    --temperature 1.0 \
    --max_new_tokens 38912 \
    --top_k -1 \
    --min_p 0 \
    --presence_penalty 0 \
    --gpu_memory_utilization 0.50 \
    --tensor_parallel_size 1 \
    --output_file "${OUT}" \
    >>"${EVAL_LOG}" 2>&1
RC=$?
echo "exit=${RC}" | tee -a "${EVAL_LOG}"
exit "${RC}"
