#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/setup/env.sh" ]]; then
  source "${REPO_ROOT}/setup/env.sh" >/dev/null
fi

: "${INFOBUY_GENERATED_DATA:?source setup/env.sh first or export INFOBUY_GENERATED_DATA}"
: "${INFOBUY_CKPT:?source setup/env.sh first or export INFOBUY_CKPT}"

TRAIN_COUNT=${TRAIN_COUNT:-16}
VAL_COUNT=${VAL_COUNT:-8}
TRAIN_SOURCE=${TRAIN_SOURCE:-${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl}
VAL_SOURCE=${VAL_SOURCE:-${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl}
SMOKE_TRAIN_RAW=${SMOKE_TRAIN_RAW:-${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_smoke.jsonl}
SMOKE_VAL_RAW=${SMOKE_VAL_RAW:-${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_smoke.jsonl}
SMOKE_TRAIN_PROTOCOL=${SMOKE_TRAIN_PROTOCOL:-${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_smoke.jsonl}
SMOKE_VAL_PROTOCOL=${SMOKE_VAL_PROTOCOL:-${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_validation_smoke.jsonl}

mkdir -p "$(dirname "${SMOKE_TRAIN_RAW}")" "$(dirname "${SMOKE_TRAIN_PROTOCOL}")"

make_slice() {
  local source_path="$1"
  local output_path="$2"
  local count="$3"
  python3 - "$source_path" "$output_path" "$count" "${REPO_ROOT}/SFT_stage/tests/fixtures/tiny_math.jsonl" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
count = int(sys.argv[3])
fallback = Path(sys.argv[4])

records = []
read_path = source if source.exists() else fallback
with read_path.open("r", encoding="utf-8") as stream:
    for line in stream:
        if line.strip():
            records.append(json.loads(line))

if not records:
    raise SystemExit(f"No records found in {read_path}")

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as stream:
    for index in range(count):
        record = dict(records[index % len(records)])
        record["id"] = f"{record.get('id', 'smoke')}_smoke_{index}"
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Wrote {count} records to {output}")
PY
}

make_slice "${TRAIN_SOURCE}" "${SMOKE_TRAIN_RAW}" "${TRAIN_COUNT}"
make_slice "${VAL_SOURCE}" "${SMOKE_VAL_RAW}" "${VAL_COUNT}"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 -m SFT_stage.build_hsp_sft \
  --input "${SMOKE_TRAIN_RAW}" \
  --output "${SMOKE_TRAIN_PROTOCOL}" \
  --emit_all_types \
  --seed 42

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 -m SFT_stage.build_hsp_sft \
  --input "${SMOKE_VAL_RAW}" \
  --output "${SMOKE_VAL_PROTOCOL}" \
  --emit_all_types \
  --seed 42

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 -m SFT_stage.preflight_hsp \
  --dataset "${SMOKE_TRAIN_PROTOCOL}" \
  --require_all_types \
  --require_context_tokens

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 -m SFT_stage.preflight_hsp \
  --dataset "${SMOKE_VAL_PROTOCOL}" \
  --require_all_types \
  --require_context_tokens

if [[ "${RUN_SFT:-0}" == "1" ]]; then
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 -m SFT_stage.train_hsp \
    --model_name "${SFT_MODEL_NAME:-Qwen/Qwen3-0.6B}" \
    --dataset "${SMOKE_TRAIN_PROTOCOL}" \
    --output_dir "${SFT_OUTPUT_DIR:-${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft-smoke}" \
    --max_seq_length "${SFT_MAX_SEQ_LENGTH:-4096}" \
    --max_train_samples "${SFT_MAX_TRAIN_SAMPLES:-32}" \
    --max_steps "${SFT_MAX_STEPS:-2}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy no
fi

if [[ "${RUN_RL:-0}" == "1" ]]; then
  MODEL_PATH=${MODEL_PATH:-${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft}
  SAVE_PATH=${SAVE_PATH:-${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_smoke}
  HSP_CONFIG=examples/config_hsp_smoke.yaml \
  MODEL_PATH="${MODEL_PATH}" \
  SAVE_PATH="${SAVE_PATH}" \
  bash "${REPO_ROOT}/RL_stage/examples/qwen3_hsp_grpo.sh"
fi

echo "Smoke data is ready:"
echo "  raw train:      ${SMOKE_TRAIN_RAW}"
echo "  raw validation: ${SMOKE_VAL_RAW}"
echo "  SFT train:      ${SMOKE_TRAIN_PROTOCOL}"
echo "  SFT validation: ${SMOKE_VAL_PROTOCOL}"
