#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${INFOBUY_PYTHON:-python}"
SPEC="configs/experiments/hsp_pilot.yaml"
MODE="--launch-tmux"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/launch_hsp_tmux.sh [SPEC] [options]
  bash scripts/launch_hsp_tmux.sh --spec configs/experiments/hsp_pilot.yaml

Defaults:
  SPEC: configs/experiments/hsp_pilot.yaml
  mode: generate run configs and start tmux queues

Options:
  --spec PATH       Experiment spec YAML.
  --dry-run         Generate configs/scripts only; do not start tmux.
  --launch-tmux     Generate configs/scripts and start tmux. Default.
  --tmux            Alias for --launch-tmux.
  --overwrite       Run even when completion markers exist.
  --gpu-pairs LIST  Semicolon-separated GPU workers, e.g. '0;1;2,3'.
  --gpus LIST       Override every run GPU assignment, e.g. '0' or '0,1'.
  --teacher-gpus LIST  GPU ids reserved by the already-running teacher service.
  --skip-teacher-check Skip teacher /generate health check.
  --teacher-check-timeout SECONDS
  --teacher-check-retries N
  --teacher-check-interval SECONDS
  -h, --help        Show this help.

Environment:
  INFOBUY_PYTHON    Python executable to use. Default: python.
  INFOBUY_GPU_PAIRS Semicolon-separated GPU workers, e.g. '0;1;2;3'.
  INFOBUY_GPUS      Override every run GPU assignment.
  INFOBUY_TEACHER_GPUS  Teacher service GPU reservation. Default in launcher: 1.
EOF
}

while (($#)); do
  case "$1" in
    --spec)
      if (($# < 2)); then
        echo "missing value for --spec" >&2
        exit 2
      fi
      SPEC="$2"
      shift 2
      ;;
    --dry-run)
      MODE=""
      shift
      ;;
    --launch-tmux|--tmux)
      MODE="--launch-tmux"
      shift
      ;;
    --skip-teacher-check)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --overwrite|--gpu-pairs|--gpus|--teacher-gpus|--teacher-check-timeout|--teacher-check-retries|--teacher-check-interval)
      if [[ "$1" == "--overwrite" ]]; then
        EXTRA_ARGS+=("$1")
        shift
      else
        if (($# < 2)); then
          echo "missing value for $1" >&2
          exit 2
        fi
        EXTRA_ARGS+=("$1" "$2")
        shift 2
      fi
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      SPEC="$1"
      shift
      ;;
  esac
done

cd "$ROOT_DIR"
source setup/env.sh >/dev/null
echo "repo: $ROOT_DIR"
echo "spec: $SPEC"
if [[ -n "$MODE" ]]; then
  echo "mode: tmux background queues"
  exec "$PYTHON_BIN" scripts/launch_hsp_experiments.py --spec "$SPEC" "$MODE" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
  echo "mode: dry run"
  exec "$PYTHON_BIN" scripts/launch_hsp_experiments.py --spec "$SPEC" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
