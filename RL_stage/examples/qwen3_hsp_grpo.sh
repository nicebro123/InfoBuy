#!/bin/bash

set -euo pipefail

export VLLM_USE_V1="${VLLM_USE_V1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_STAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${RL_STAGE_DIR}/.." && pwd)"

if [[ -z "${HSP_CKPT:-}" && -f "${REPO_ROOT}/setup/env.sh" ]]; then
    source "${REPO_ROOT}/setup/env.sh" >/dev/null
fi

CKPT_ROOT=${INFOBUY_CKPT:-${HSP_CKPT:-"${REPO_ROOT}/checkpoints"}}
MODEL_PATH=${MODEL_PATH:-${CKPT_ROOT}/sft/qwen3-0.6b-hsp-sft}
HSP_CONFIG=${HSP_CONFIG:-examples/config_hsp.yaml}
if [[ "${HSP_CONFIG}" != /* && ! -f "${RL_STAGE_DIR}/${HSP_CONFIG}" && -f "${REPO_ROOT}/${HSP_CONFIG}" ]]; then
    HSP_CONFIG="${REPO_ROOT}/${HSP_CONFIG}"
fi
DEFAULT_EXPERIMENT_NAME=qwen3_hsp_grpo_main
if [[ "${HSP_CONFIG}" == *"config_hsp_shaped.yaml" ]]; then
    DEFAULT_EXPERIMENT_NAME=qwen3_hsp_grpo_shaped
fi
SAVE_PATH=${SAVE_PATH:-${CKPT_ROOT}/rl/${DEFAULT_EXPERIMENT_NAME}}

cd "${RL_STAGE_DIR}"

PYTHONPATH="${REPO_ROOT}:${RL_STAGE_DIR}:${PYTHONPATH:-}" python3 "${REPO_ROOT}/SFT_stage/preflight_hsp.py" \
    --model_path "${MODEL_PATH}" \
    --rl_config "${HSP_CONFIG}" \
    --require_context_tokens

PYTHONPATH="${REPO_ROOT}:${RL_STAGE_DIR}:${PYTHONPATH:-}" python3 -m verl.trainer.main \
    config="${HSP_CONFIG}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    trainer.save_checkpoint_path="${SAVE_PATH}" \
    "$@"
