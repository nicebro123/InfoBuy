#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${INFOBUY_PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage:
  bash run.sh <command> [options]

Core commands:
  setup                 Create external store directories.
  download-models       Download Qwen3 student/teacher weights.
  download-data         Download/cache source and eval datasets.
  build-data            Build the default HSP pilot dataset.
  smoke                 Build and preflight tiny smoke data.
  sft-smoke             Run a tiny SFT smoke in tmux unless --foreground.
  sft                   Run full HSP SFT in tmux unless --foreground.
  teacher               Start the teacher service in tmux unless --foreground.
  rl-smoke              Launch the RL smoke spec in tmux.
  experiment NAME       Run one legacy named experiment in tmux.
  train                 Launch official config-driven experiment specs.
  eval                  Run HSP batch evaluation.
  checks                Run local tests and shell syntax checks.

Common options:
  --gpu ID              GPU id for teacher/SFT/RL/eval. Default varies by command.
  --port PORT           Teacher service port. Default: 7778.
  --spec PATH           Experiment spec for train/rl-smoke.
  --gpu-pairs LIST      Semicolon-separated GPU workers for train, e.g. '0;1;2;3'.
  --overwrite           Run experiment queues even when completion markers exist.
  --skip-smoke          For train: do not run smoke checks before materializing specs.
  --foreground          Run long command in the current shell instead of tmux.
  --dry-run             Generate experiment scripts without starting tmux.
  -h, --help            Show this help.

Examples:
  bash run.sh smoke
  bash run.sh teacher --gpu 1
  bash run.sh sft --gpu 0
  bash run.sh rl-smoke --gpu 0
  bash run.sh train --gpu-pairs '0;1;2;3'
EOF
}

start_tmux_or_foreground() {
  local session="$1"
  local command="$2"
  local foreground="$3"
  if [[ "$foreground" == "1" ]]; then
    bash -lc "$command"
    return
  fi
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed; rerun with --foreground or install tmux." >&2
    exit 2
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session"
  else
    tmux new-session -d -s "$session" "$command"
    echo "started tmux session: $session"
    echo "attach: tmux attach -t $session"
  fi
}

require_env() {
  source "$ROOT_DIR/setup/env.sh" >/dev/null
}

command=${1:-}
if [[ -z "$command" || "$command" == "-h" || "$command" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

gpu=""
port=7778
spec=""
gpu_pairs=""
foreground=0
dry_run=0
overwrite=0
skip_smoke=0
extra=()

while (($#)); do
  case "$1" in
    --gpu)
      gpu="$2"
      shift 2
      ;;
    --port)
      port="$2"
      shift 2
      ;;
    --spec)
      spec="$2"
      shift 2
      ;;
    --gpu-pairs)
      gpu_pairs="$2"
      shift 2
      ;;
    --foreground)
      foreground=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    --skip-smoke)
      skip_smoke=1
      shift
      ;;
    --)
      shift
      extra+=("$@")
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      extra+=("$1")
      shift
      ;;
  esac
done

cd "$ROOT_DIR"

case "$command" in
  setup)
    require_env
    bash setup/make_dirs.sh
    ;;
  download-models)
    require_env
    bash setup/download_models.sh
    ;;
  download-data)
    require_env
    bash setup/download_data.sh
    ;;
  build-data)
    require_env
    "$PYTHON_BIN" build_and_upload_hsp_dataset.py --build_only "${extra[@]+"${extra[@]}"}"
    ;;
  smoke)
    require_env
    bash experiments/run_hsp_smoke.sh
    ;;
  sft-smoke)
    require_env
    gpu="${gpu:-0}"
    cmd="cd $(printf '%q' "$ROOT_DIR") && source setup/env.sh >/dev/null && CUDA_VISIBLE_DEVICES=$(printf '%q' "$gpu") RUN_SFT=1 SFT_MODEL_NAME=\${INFOBUY_PRETRAINED_MODELS}/Qwen3-0.6B bash experiments/run_hsp_smoke.sh"
    start_tmux_or_foreground "infobuy_sft_smoke_g${gpu//,/}" "$cmd" "$foreground"
    ;;
  sft)
    require_env
    gpu="${gpu:-0}"
    cmd="cd $(printf '%q' "$ROOT_DIR") && source setup/env.sh >/dev/null && CUDA_VISIBLE_DEVICES=$(printf '%q' "$gpu") $PYTHON_BIN -m SFT_stage.train_hsp --model_name \${INFOBUY_PRETRAINED_MODELS}/Qwen3-0.6B --dataset \${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl --output_dir \${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft --max_seq_length 12288 --learning_rate 5e-6 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --num_train_epochs 1 --bf16"
    start_tmux_or_foreground "infobuy_sft_g${gpu//,/}" "$cmd" "$foreground"
    ;;
  teacher)
    require_env
    gpu="${gpu:-1}"
    cmd="cd $(printf '%q' "$ROOT_DIR") && source setup/env.sh >/dev/null && CUDA_VISIBLE_DEVICES=$(printf '%q' "$gpu") $PYTHON_BIN utils/vllm_service.py --model_path \${INFOBUY_TEACHER_MODELS}/qwen3-8b-main --port $(printf '%q' "$port") --tensor_parallel_size 1 --trust_remote_code"
    start_tmux_or_foreground "infobuy_teacher_${port}" "$cmd" "$foreground"
    ;;
  rl-smoke)
    require_env
    gpu="${gpu:-0}"
    spec="${spec:-configs/experiments/hsp_smoke.yaml}"
    args=(--spec "$spec" --gpus "$gpu")
    if ((dry_run)); then
      args+=(--dry-run)
    else
      args+=(--launch-tmux)
    fi
    if ((overwrite)); then
      args+=(--overwrite)
    fi
    "$PYTHON_BIN" scripts/launch_hsp_experiments.py "${args[@]}"
    ;;
  experiment)
    require_env
    name=${extra[0]:-}
    if [[ -z "$name" ]]; then
      echo "Usage: bash run.sh experiment NAME [--gpu ID] [--foreground]" >&2
      exit 2
    fi
    gpu="${gpu:-0}"
    cmd="cd $(printf '%q' "$ROOT_DIR") && source setup/env.sh >/dev/null && CUDA_VISIBLE_DEVICES=$(printf '%q' "$gpu") bash experiments/run_hsp_experiment.sh $(printf '%q' "$name")"
    start_tmux_or_foreground "infobuy_hsp_${name}_g${gpu//,/}" "$cmd" "$foreground"
    ;;
  train)
    require_env
    args=()
    if [[ -n "$spec" ]]; then
      args+=(--spec "$spec")
    fi
    if [[ -n "$gpu_pairs" ]]; then
      args+=(--gpu-pairs "$gpu_pairs")
    fi
    if ((dry_run)); then
      args+=(--dry-run)
    fi
    if ((overwrite)); then
      args+=(--overwrite)
    fi
    if ((skip_smoke)); then
      args+=(--skip-smoke)
    fi
    bash scripts/launch_all_hsp_experiments.sh "${args[@]+"${args[@]}"}"
    ;;
  eval)
    require_env
    gpu="${gpu:-0}"
    export SKIP_LLM_RECHECK="${SKIP_LLM_RECHECK:-1}"
    CUDA_VISIBLE_DEVICES="$gpu" bash eval/evaluate_forhelp.bash \
      "${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft" \
      Qwen3-8B \
      "$port" \
      "$gpu" \
      hsp \
      8
    ;;
  checks)
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m unittest discover -s . -p 'test*.py'
    "$PYTHON_BIN" - <<'PY'
from pathlib import Path
path = Path("scripts/launch_hsp_experiments.py")
compile(path.read_text(encoding="utf-8"), str(path), "exec")
PY
    bash -n \
      run.sh \
      setup/env.sh \
      setup/make_dirs.sh \
      setup/link_data.sh \
      setup/download_models.sh \
      setup/download_data.sh \
      experiments/run_hsp_smoke.sh \
      experiments/run_hsp_experiment.sh \
      scripts/launch_hsp_tmux.sh \
      scripts/launch_all_hsp_experiments.sh \
      eval/evaluate_forhelp.bash \
      RL_stage/examples/qwen3_hsp_grpo.sh
    git diff --check
    ;;
  *)
    echo "unknown command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
