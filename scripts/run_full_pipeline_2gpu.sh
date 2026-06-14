#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_full_pipeline_2gpu.sh [--launch] [--overwrite] [--skip-sft]

Prepare the full InfoBuy HSP experiment pipeline for the minimum two-GPU
layout. By default this only materializes run_config.yaml files and writes a
teacher queue plus one worker queue. Pass --launch to start tmux workers:

  infobuy_full_pipeline_teacher_g0
  infobuy_full_pipeline_worker_g1

Environment:
  PYTHON=/path/to/python        Python executable, default: $INFOBUY_PYTHON or python
  SESSION_PREFIX=name          tmux session prefix, default: infobuy_full_pipeline
  QUEUE_ROOT=path              Queue root, default: $INFOBUY_STORE/experiments/full_pipeline_2gpu
  TEACHER_GPU=0                Teacher GPU. Default: 0
  WORKER_GPU=1                 Student worker GPU. Default: 1
  TEACHER_PORT=7778            Teacher service port. Default: 7778

The teacher GPU and worker GPU must be distinct.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source setup/env.sh >/dev/null

PYTHON_BIN="${PYTHON:-${INFOBUY_PYTHON:-python}}"
SESSION_PREFIX="${SESSION_PREFIX:-infobuy_full_pipeline}"
QUEUE_ROOT="${QUEUE_ROOT:-${INFOBUY_STORE}/experiments/full_pipeline_2gpu}"
TEACHER_GPU="${TEACHER_GPU:-0}"
WORKER_GPU="${WORKER_GPU:-1}"
TEACHER_PORT="${TEACHER_PORT:-7778}"
LAUNCH=0
OVERWRITE=0
SKIP_SFT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launch)
      LAUNCH=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --skip-sft)
      SKIP_SFT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$TEACHER_GPU" == "$WORKER_GPU" ]]; then
  echo "TEACHER_GPU and WORKER_GPU must be distinct for the two-GPU pipeline." >&2
  exit 2
fi

SPECS=(
  configs/experiments/hsp_official.yaml
  configs/experiments/hsp_analysis.yaml
)

mkdir -p "$QUEUE_ROOT"
TEACHER_QUEUE="$QUEUE_ROOT/run_teacher_gpu${TEACHER_GPU}.sh"
WORKER_QUEUE="$QUEUE_ROOT/run_worker_gpu${WORKER_GPU}.sh"

cat > "$TEACHER_QUEUE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$ROOT")
export INFOBUY_STORE=$(printf '%q' "$INFOBUY_STORE")
source setup/env.sh >/dev/null
echo "[full-pipeline] teacher GPU ${TEACHER_GPU} starting at \$(date)"
bash run.sh teacher --gpu $(printf '%q' "$TEACHER_GPU") --port $(printf '%q' "$TEACHER_PORT")
echo "[full-pipeline] teacher command returned at \$(date)"
EOF

cat > "$WORKER_QUEUE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$ROOT")
export INFOBUY_STORE=$(printf '%q' "$INFOBUY_STORE")
source setup/env.sh >/dev/null
export INFOBUY_TEACHER_GPUS=$(printf '%q' "$TEACHER_GPU")
export INFOBUY_GPUS=$(printf '%q' "$WORKER_GPU")
echo "[full-pipeline] worker GPU ${WORKER_GPU} queue started at \$(date)"

echo "[full-pipeline] full-data preflight"
$(printf '%q' "$PYTHON_BIN") scripts/check_hsp_full_data.py --strict

if [[ "$SKIP_SFT" != "1" ]]; then
  echo "[full-pipeline] protocol SFT"
  bash run.sh sft --gpu $(printf '%q' "$WORKER_GPU")
  echo "[full-pipeline] token probe"
  bash run.sh token-probe --gpu $(printf '%q' "$WORKER_GPU")
else
  echo "[full-pipeline] skip protocol SFT/token probe"
fi

EOF

for spec in "${SPECS[@]}"; do
  echo "[prepare] $spec"
  prepare_args=(
    scripts/launch_hsp_experiments.py
    --spec "$spec"
    --gpus "$WORKER_GPU"
    --teacher-gpus "$TEACHER_GPU"
    --skip-teacher-check
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    prepare_args+=(--overwrite)
  fi
  "$PYTHON_BIN" "${prepare_args[@]}"

  study_dir="$("$PYTHON_BIN" - "$spec" <<'PY'
from pathlib import Path
import os
import sys
import yaml

def slug(value: str) -> str:
    chars = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_", ".", "+"}:
            chars.append(char)
        elif char.isspace() or char in {",", "/", ":"}:
            chars.append("_")
    return "".join(chars).strip("._-") or "run"

spec_path = Path(sys.argv[1])
spec = yaml.safe_load(spec_path.read_text()) or {}
root = Path(os.path.expandvars(str(spec.get("output_root", "${INFOBUY_STORE}/experiments")))).expanduser()
if not root.is_absolute():
    root = (Path.cwd() / root).resolve()
print(root / slug(str(spec.get("study_name") or spec_path.stem)))
PY
)"
  study_script="$study_dir/run_gpu${WORKER_GPU}.sh"
  if [[ -x "$study_script" ]]; then
    cat >> "$WORKER_QUEUE" <<EOF
echo "[full-pipeline] running $(printf '%q' "$study_script") at \$(date)"
bash $(printf '%q' "$study_script")

EOF
  else
    echo "expected study queue not found or not executable: $study_script" >&2
    exit 1
  fi
done

cat >> "$WORKER_QUEUE" <<EOF
echo "[full-pipeline] worker queue finished at \$(date)"
EOF

chmod +x "$TEACHER_QUEUE" "$WORKER_QUEUE"
echo "[ready] teacher queue: $TEACHER_QUEUE"
echo "[ready] worker queue:  $WORKER_QUEUE"

if [[ "$LAUNCH" != "1" ]]; then
  echo "[ready] launch with: bash scripts/run_full_pipeline_2gpu.sh --launch"
  exit 0
fi

teacher_session="${SESSION_PREFIX}_teacher_g${TEACHER_GPU}"
worker_session="${SESSION_PREFIX}_worker_g${WORKER_GPU}"
for session in "$teacher_session" "$worker_session"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

tmux new-session -d -s "$teacher_session" "bash $(printf '%q' "$TEACHER_QUEUE")"
tmux new-session -d -s "$worker_session" "bash $(printf '%q' "$WORKER_QUEUE")"
echo "[launch] $teacher_session -> $TEACHER_QUEUE"
echo "[launch] $worker_session -> $WORKER_QUEUE"
