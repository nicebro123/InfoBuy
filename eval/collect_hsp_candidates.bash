#!/bin/bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: bash eval/collect_hsp_candidates.bash <student_model> <dataset> <teacher_name> <port> [samples_per_mode] [generator_args...]" >&2
  exit 2
fi

student_model=$1
dataset=$2
teacher_name=$3
port=$4

case "${dataset}" in
  math|gsm8k|amc|minerva|olympiad|aime2024|aime2025|mmlu_pro|bbeh|super_gpqa|gpqa)
    echo "Dataset '${dataset}' is wired to evaluation data and cannot be used to build replay SFT." >&2
    echo "Use a training source such as dataset=local_json with --name /path/to/train.jsonl." >&2
    exit 2
    ;;
esac

if [ "$#" -ge 5 ]; then
  samples_per_mode=$5
  shift 5
else
  samples_per_mode=4
  shift 4
fi

for collection_mode in independent force_ask_first force_verify_after_draft policy; do
  python -m eval.generate_withhelp \
    --small_model "${student_model}" \
    --dataset "${dataset}" \
    --larger_model "${teacher_name}" \
    --large_model_url "http://127.0.0.1:${port}/generate" \
    --interaction_policy hsp \
    --data_role train \
    --collection_mode "${collection_mode}" \
    --samples_per_question "${samples_per_mode}" \
    "$@"
done
