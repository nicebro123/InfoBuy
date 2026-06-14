#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
source setup/env.sh >/dev/null

SPEC="${SPEC:-configs/experiments/hsp_official.yaml}"
TEACHER_GPU="${TEACHER_GPU:-0}"
WORKER_GPUS="${WORKER_GPUS:-1}"
PORT="${PORT:-7778}"
PYTHON_BIN="${INFOBUY_PYTHON:-python}"

usage() {
  cat >&2 <<USAGE
Usage: TEACHER_GPU=0 WORKER_GPUS=1 bash scripts/launch_hsp_full_stack.sh [launcher args]

Environment:
  SPEC          Experiment spec. Default: configs/experiments/hsp_official.yaml
  TEACHER_GPU   GPU for Qwen3-8B teacher service. Default: 0
  WORKER_GPUS   Worker GPU queue, e.g. "1" or "1;2;3". Default: 1
  PORT          Teacher service port. Default: 7778

This is the minimum two-GPU full experiment launcher:
  GPU 0: teacher service
  GPU 1: sequential worker queue
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

IFS=';' read -r -a worker_gpu_array <<< "$WORKER_GPUS"
for gpu_id in "${worker_gpu_array[@]}"; do
  if [[ "$gpu_id" == "$TEACHER_GPU" ]]; then
    echo "TEACHER_GPU must not also appear in WORKER_GPUS: $TEACHER_GPU" >&2
    exit 2
  fi
done

echo "==> Full-data preflight"
"$PYTHON_BIN" scripts/check_hsp_full_data.py --strict

teacher_healthy() {
  curl -sS --max-time 8 \
    -H 'Content-Type: application/json' \
    -d '[{"prompt":"ping","max_tokens":4}]' \
    "http://127.0.0.1:${PORT}/generate" >/dev/null
}

if teacher_healthy; then
  echo "==> Teacher already healthy on port ${PORT}"
else
  echo "==> Starting teacher on GPU ${TEACHER_GPU}, port ${PORT}"
  bash run.sh teacher --gpu "$TEACHER_GPU" --port "$PORT"
fi

echo "==> Waiting for teacher health"
for attempt in $(seq 1 60); do
  if teacher_healthy; then
    echo "==> Teacher health check passed"
    break
  fi
  if [[ "$attempt" == "60" ]]; then
    echo "Teacher did not become healthy on port ${PORT}" >&2
    exit 1
  fi
  sleep 5
done

echo "==> Launching full experiment queue"
echo "    spec:        ${SPEC}"
echo "    teacher GPU: ${TEACHER_GPU}"
echo "    worker GPUs: ${WORKER_GPUS}"

"$PYTHON_BIN" scripts/launch_hsp_experiments.py \
  --spec "$SPEC" \
  --gpus "$WORKER_GPUS" \
  --teacher-gpus "$TEACHER_GPU" \
  --launch \
  "$@"
