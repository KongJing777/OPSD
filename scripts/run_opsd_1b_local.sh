#!/usr/bin/env bash
set -euo pipefail

# Local reproduction launcher. Defaults to the public Hub id
# Qwen/Qwen3-1.7B (see README). Override MODEL_NAME_OR_PATH to a local snapshot:
#   export MODEL_NAME_OR_PATH=./models/Qwen3-1.7B
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/runs}"
RUN_CONFIG="${RUN_CONFIG:-qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005_local}"
# Safe defaults for a single idle GPU. Override both variables together for a multi-GPU run.
CUDA_DEVICES="${CUDA_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAX_STEPS="${MAX_STEPS:-100}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_LENGTH="${MAX_LENGTH:-20000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
OUTCOME_GATE_LAMBDA="${OUTCOME_GATE_LAMBDA:-0}"
OUTCOME_GATE_MODE="${OUTCOME_GATE_MODE:-all_error}"

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate)}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${REPO_DIR}/accelerate_local.yaml}"

# Optional cache isolation. Unset HF_HOME to use the default ~/.cache/huggingface.
if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HOME
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
fi

cd "${REPO_DIR}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${ACCELERATE_BIN}" launch \
    --config_file "${ACCELERATE_CONFIG_FILE}" \
    --num_processes "${NUM_PROCESSES}" \
    --gradient_accumulation_steps 2 \
    --main_process_port "${MAIN_PROCESS_PORT:-12949}" \
    opsd_train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir "${OUTPUT_DIR}" \
    --run_config "${RUN_CONFIG}" \
    --num_train_epochs 30 \
    --max_steps "${MAX_STEPS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --torch_dtype bfloat16 \
    --max_length "${MAX_LENGTH}" \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.30}" \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --outcome_gate_lambda "${OUTCOME_GATE_LAMBDA}" \
    --outcome_gate_mode "${OUTCOME_GATE_MODE}" \
    --wandb_project "${WANDB_PROJECT:-OPSD}"
