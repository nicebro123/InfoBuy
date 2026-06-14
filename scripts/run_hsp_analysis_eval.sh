#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
source setup/env.sh >/dev/null

MODEL_PATH="${MODEL_PATH:-${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft}"
TEACHER_NAME="${TEACHER_NAME:-Qwen3-8B}"
PORT="${PORT:-7778}"
GPU_QUEUE="${GPU_QUEUE:-1}"
SAMPLES_PER_QUESTION="${SAMPLES_PER_QUESTION:-8}"
PYTHON_BIN="${INFOBUY_PYTHON:-python}"
BASH_BIN="${BASH_BIN:-bash}"

FULL_TASKS="${FULL_TASKS:-math gsm8k minerva olympiad aime2024 aime2025}"
OOD_TASKS="${OOD_TASKS:-bbeh mmlu_pro super_gpqa}"

usage() {
  cat >&2 <<USAGE
Usage: MODEL_PATH=/path/to/model GPU_QUEUE=1 bash scripts/run_hsp_analysis_eval.sh [suite]

Suites:
  full            Full six-benchmark HSP policy eval.
  teacher-free    Independent/no-help eval; no teacher calls should be used.
  forced          Forced ASK and forced VERIFY collection evals.
  fixed-budget    Fixed small/mid/large ASK/VERIFY budget evals.
  ood             OOD eval on BBEH, MMLU-Pro, and SuperGPQA.
  all             Run all suites above. Default: all.

Environment:
  MODEL_PATH              Student model or merged RL checkpoint.
  TEACHER_NAME            Teacher label for result paths. Default: Qwen3-8B.
  PORT                    Teacher service port. Default: 7778.
  GPU_QUEUE               Space-separated eval GPU queue. Default: 1.
  SAMPLES_PER_QUESTION    Samples per question. Default: 8.
  FULL_TASKS              Full math task list.
  OOD_TASKS               OOD task list.

Do not set MAX_EXAMPLES for paper-facing full evaluation.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

suite="${1:-all}"
export SKIP_LLM_RECHECK="${SKIP_LLM_RECHECK:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

run_eval() {
  local output_tag="$1"
  local tasks="$2"
  local collection_mode="$3"
  shift 3

  echo "==> suite=${output_tag} tasks=[${tasks}] mode=${collection_mode}"
  EVAL_TASKS="$tasks" OUTPUT_TAG="$output_tag" "$@" \
    "$BASH_BIN" eval/evaluate_forhelp.bash \
      "$MODEL_PATH" \
      "$TEACHER_NAME" \
      "$PORT" \
      "$GPU_QUEUE" \
      hsp \
      "$SAMPLES_PER_QUESTION" \
      "$collection_mode"
}

case "$suite" in
  full)
    run_eval full_policy "$FULL_TASKS" policy env
    ;;
  teacher-free)
    run_eval teacher_free_independent "$FULL_TASKS" independent env MAX_INTERACTIONS=0
    ;;
  forced)
    run_eval forced_ask_first "$FULL_TASKS" force_ask_first env
    run_eval forced_verify_after_draft "$FULL_TASKS" force_verify_after_draft env
    ;;
  fixed-budget)
    run_eval fixed_budget_32_32 "$FULL_TASKS" policy env ASK_BUDGET_TOKENS=32 VERIFY_BUDGET_TOKENS=32
    run_eval fixed_budget_64_96 "$FULL_TASKS" policy env ASK_BUDGET_TOKENS=64 VERIFY_BUDGET_TOKENS=96
    run_eval fixed_budget_128_192 "$FULL_TASKS" policy env ASK_BUDGET_TOKENS=128 VERIFY_BUDGET_TOKENS=192
    ;;
  ood)
    run_eval ood_policy "$OOD_TASKS" policy env
    run_eval ood_teacher_free_independent "$OOD_TASKS" independent env MAX_INTERACTIONS=0
    ;;
  all)
    "$0" full
    "$0" teacher-free
    "$0" forced
    "$0" fixed-budget
    "$0" ood
    ;;
  *)
    echo "Unknown suite: $suite" >&2
    usage
    exit 2
    ;;
esac
