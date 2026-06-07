#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/setup/env.sh" ]]; then
  source "${REPO_ROOT}/setup/env.sh" >/dev/null
fi

: "${INFOBUY_CKPT:?source setup/env.sh first or export INFOBUY_CKPT}"

EXPERIMENT=${1:-}
if [[ -z "${EXPERIMENT}" ]]; then
  echo "Usage: bash experiments/run_hsp_experiment.sh <smoke|main|shaped|no_cost|cost_low|cost_high|trust_low|trust_high|budget_small|budget_large|interactions_1|interactions_4|kl_low|kl_high|lr_low|lr_high> [extra_overrides...]" >&2
  exit 2
fi
shift || true

HSP_CONFIG=${HSP_CONFIG:-examples/config_hsp.yaml}
RUN_NAME=${EXPERIMENT}
overrides=()

set_pair_weight() {
  local key="$1"
  local value="$2"
  overrides+=(
    "worker.reward.reward_function_kwargs.${key}=${value}"
    "worker.val_reward.reward_function_kwargs.${key}=${value}"
  )
}

case "${EXPERIMENT}" in
  smoke)
    HSP_CONFIG=examples/config_hsp_smoke.yaml
    RUN_NAME=qwen3_hsp_grpo_smoke
    ;;
  main)
    HSP_CONFIG=examples/config_hsp.yaml
    RUN_NAME=qwen3_hsp_grpo_main
    ;;
  shaped)
    HSP_CONFIG=examples/config_hsp_shaped.yaml
    RUN_NAME=qwen3_hsp_grpo_shaped
    ;;
  no_cost)
    RUN_NAME=qwen3_hsp_grpo_no_cost
    set_pair_weight teacher_cost_weight 0.0
    ;;
  cost_low)
    RUN_NAME=qwen3_hsp_grpo_cost_005
    set_pair_weight teacher_cost_weight 0.05
    ;;
  cost_high)
    RUN_NAME=qwen3_hsp_grpo_cost_030
    set_pair_weight teacher_cost_weight 0.30
    ;;
  trust_low)
    RUN_NAME=qwen3_hsp_grpo_trust_025
    set_pair_weight wrong_accept_weight 0.25
    set_pair_weight wrong_reject_weight 0.25
    set_pair_weight wrong_implicit_adoption_weight 0.25
    ;;
  trust_high)
    RUN_NAME=qwen3_hsp_grpo_trust_080
    set_pair_weight wrong_accept_weight 0.80
    set_pair_weight wrong_reject_weight 0.80
    set_pair_weight wrong_implicit_adoption_weight 0.80
    ;;
  budget_small)
    RUN_NAME=qwen3_hsp_grpo_budget_small
    overrides+=(
      worker.rollout.ask_budget_tokens=32
      worker.rollout.verify_budget_tokens=48
      worker.reward.reward_function_kwargs.teacher_token_budget=80.0
      worker.val_reward.reward_function_kwargs.teacher_token_budget=80.0
    )
    ;;
  budget_large)
    RUN_NAME=qwen3_hsp_grpo_budget_large
    overrides+=(
      worker.rollout.ask_budget_tokens=96
      worker.rollout.verify_budget_tokens=128
      worker.reward.reward_function_kwargs.teacher_token_budget=224.0
      worker.val_reward.reward_function_kwargs.teacher_token_budget=224.0
    )
    ;;
  interactions_1)
    RUN_NAME=qwen3_hsp_grpo_interactions_1
    overrides+=(worker.rollout.max_interactions=1)
    ;;
  interactions_4)
    RUN_NAME=qwen3_hsp_grpo_interactions_4
    overrides+=(worker.rollout.max_interactions=4)
    ;;
  kl_low)
    RUN_NAME=qwen3_hsp_grpo_kl_003
    overrides+=(algorithm.kl_coef=0.003)
    ;;
  kl_high)
    RUN_NAME=qwen3_hsp_grpo_kl_030
    overrides+=(algorithm.kl_coef=0.03)
    ;;
  lr_low)
    RUN_NAME=qwen3_hsp_grpo_lr_3e-7
    overrides+=(worker.actor.optim.lr=3.0e-7)
    ;;
  lr_high)
    RUN_NAME=qwen3_hsp_grpo_lr_3e-6
    overrides+=(worker.actor.optim.lr=3.0e-6)
    ;;
  *)
    echo "Unknown experiment: ${EXPERIMENT}" >&2
    exit 2
    ;;
esac

MODEL_PATH=${MODEL_PATH:-${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft}
SAVE_PATH=${SAVE_PATH:-${INFOBUY_CKPT}/rl/${RUN_NAME}}

HSP_CONFIG="${HSP_CONFIG}" \
MODEL_PATH="${MODEL_PATH}" \
SAVE_PATH="${SAVE_PATH}" \
bash "${REPO_ROOT}/RL_stage/examples/qwen3_hsp_grpo.sh" \
  trainer.experiment_name="${RUN_NAME}" \
  "${overrides[@]}" \
  "$@"
