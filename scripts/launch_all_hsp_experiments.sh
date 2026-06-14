#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${INFOBUY_PYTHON:-python}"
TEACHER_GPUS="${INFOBUY_TEACHER_GPUS:-1}"
RUN_SMOKE=1
LAUNCH_TMUX=1
EXTRA_ARGS=()
SPECS=()
DEFAULT_SPECS=(
  "configs/experiments/hsp_pilot.yaml"
  "configs/experiments/hsp_ablation_cost.yaml"
  "configs/experiments/hsp_ablation_trust.yaml"
  "configs/experiments/hsp_ablation_budget.yaml"
  "configs/experiments/hsp_ablation_interactions.yaml"
  "configs/experiments/hsp_hparam_sweep.yaml"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/launch_all_hsp_experiments.sh [options]

Default behavior:
  1. run smoke data/protocol checks;
  2. materialize each official experiment spec;
  3. start one tmux queue per spec/GPU worker.

Options:
  --spec PATH            Add one experiment spec. Repeatable.
  --gpu-pairs LIST       Semicolon-separated GPU workers, e.g. '0;1;2;3'.
  --teacher-gpus LIST    GPU ids reserved by the already-running teacher service.
  --dry-run              Generate configs/scripts only; do not start tmux.
  --launch-tmux, --tmux  Start tmux queues. Default.
  --overwrite            Run even when completion markers exist.
  --skip-teacher-check   Skip teacher /generate health check.
  --teacher-check-timeout SECONDS
  --teacher-check-retries N
  --teacher-check-interval SECONDS
  --skip-smoke           Do not run experiments/run_hsp_smoke.sh first.
  -h, --help             Show this help.

Environment:
  INFOBUY_PYTHON         Python executable to use. Default: python.
  INFOBUY_GPU_PAIRS      Semicolon-separated GPU workers. CLI overrides it.
  INFOBUY_TEACHER_GPUS   Teacher service GPU reservation. Default: 1.
  INFOBUY_TMUX_SESSION_PREFIX  Optional tmux session prefix.
EOF
}

while (($#)); do
  case "$1" in
    --spec)
      if (($# < 2)); then
        echo "missing value for --spec" >&2
        exit 2
      fi
      SPECS+=("$2")
      shift 2
      ;;
    --gpu-pairs)
      if (($# < 2)); then
        echo "missing value for --gpu-pairs" >&2
        exit 2
      fi
      export INFOBUY_GPU_PAIRS="$2"
      shift 2
      ;;
    --teacher-gpus)
      if (($# < 2)); then
        echo "missing value for --teacher-gpus" >&2
        exit 2
      fi
      TEACHER_GPUS="$2"
      shift 2
      ;;
    --dry-run)
      LAUNCH_TMUX=0
      shift
      ;;
    --launch-tmux|--tmux)
      LAUNCH_TMUX=1
      shift
      ;;
    --overwrite)
      EXTRA_ARGS+=("--overwrite")
      shift
      ;;
    --skip-teacher-check)
      EXTRA_ARGS+=("--skip-teacher-check")
      shift
      ;;
    --teacher-check-timeout|--teacher-check-retries|--teacher-check-interval)
      if (($# < 2)); then
        echo "missing value for $1" >&2
        exit 2
      fi
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --skip-smoke)
      RUN_SMOKE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while (($#)); do
        SPECS+=("$1")
        shift
      done
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      SPECS+=("$1")
      shift
      ;;
  esac
done

if ((${#SPECS[@]} == 0)); then
  SPECS=("${DEFAULT_SPECS[@]}")
fi

cd "$ROOT_DIR"
source setup/env.sh >/dev/null

if ((LAUNCH_TMUX)) && [[ -z "${INFOBUY_GPU_PAIRS:-}" && -z "${HSP_GPU_PAIRS:-}" ]]; then
  echo "HSP RL launch requires explicit training GPUs via --gpu-pairs." >&2
  echo "Example for a two-GPU machine with teacher on GPU 1:" >&2
  echo "  bash scripts/launch_all_hsp_experiments.sh --teacher-gpus 1 --gpu-pairs '0'" >&2
  exit 2
fi

echo "repo: $ROOT_DIR"
if ((LAUNCH_TMUX)); then
  echo "mode: tmux background queues"
else
  echo "mode: dry run"
fi
echo "specs:"
for spec in "${SPECS[@]}"; do
  echo "  - $spec"
done

if ((RUN_SMOKE)); then
  bash experiments/run_hsp_smoke.sh
fi

for spec in "${SPECS[@]}"; do
  if ((LAUNCH_TMUX)); then
    "$PYTHON_BIN" scripts/launch_hsp_experiments.py \
      --spec "$spec" \
      --launch-tmux \
      --teacher-gpus "$TEACHER_GPUS" \
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
  else
    "$PYTHON_BIN" scripts/launch_hsp_experiments.py \
      --spec "$spec" \
      --teacher-gpus "$TEACHER_GPUS" \
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
  fi
done
