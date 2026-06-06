#!/usr/bin/env bash
# Create the InfoBuy external storage tree.
# Usage: source setup/env.sh && bash setup/make_dirs.sh
set -euo pipefail
: "${INFOBUY_STORE:?source setup/env.sh first (or export INFOBUY_STORE)}"
: "${HF_HOME:?source setup/env.sh first}"
: "${HF_HUB_CACHE:?source setup/env.sh first}"
: "${HF_DATASETS_CACHE:?source setup/env.sh first}"
: "${INFOBUY_PRETRAINED_MODELS:?source setup/env.sh first}"
: "${INFOBUY_TEACHER_MODELS:?source setup/env.sh first}"
: "${INFOBUY_HF_DOWNLOADS:?source setup/env.sh first}"
: "${INFOBUY_BENCHMARKS:?source setup/env.sh first}"
: "${INFOBUY_GENERATED_DATA:?source setup/env.sh first}"
: "${INFOBUY_CKPT:?source setup/env.sh first}"
: "${STORAGE_PATH:?source setup/env.sh first}"
: "${INFOBUY_LOGS:?source setup/env.sh first}"
: "${INFOBUY_SERVICES:?source setup/env.sh first}"
: "${INFOBUY_TMP:?source setup/env.sh first}"
: "${INFOBUY_BACKUPS:?source setup/env.sh first}"

mkdir -p \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$INFOBUY_PRETRAINED_MODELS" \
  "$INFOBUY_TEACHER_MODELS" \
  "$INFOBUY_HF_DOWNLOADS/AI-MO__NuminaMath-CoT" \
  "$INFOBUY_HF_DOWNLOADS/optional_baselines" \
  "$INFOBUY_BENCHMARKS/math500" \
  "$INFOBUY_BENCHMARKS/gsm8k" \
  "$INFOBUY_BENCHMARKS/aime2024" \
  "$INFOBUY_BENCHMARKS/aime2025" \
  "$INFOBUY_BENCHMARKS/amc23" \
  "$INFOBUY_BENCHMARKS/minerva" \
  "$INFOBUY_BENCHMARKS/olympiadbench" \
  "$INFOBUY_GENERATED_DATA/raw" \
  "$INFOBUY_GENERATED_DATA/protocol" \
  "$INFOBUY_GENERATED_DATA/flat" \
  "$INFOBUY_GENERATED_DATA/replay" \
  "$INFOBUY_GENERATED_DATA/trust" \
  "$INFOBUY_GENERATED_DATA/purchase" \
  "$INFOBUY_GENERATED_DATA/manifests" \
  "$INFOBUY_GENERATED_DATA/splits" \
  "$INFOBUY_CKPT/sft" \
  "$INFOBUY_CKPT/rl" \
  "$INFOBUY_CKPT/merged" \
  "$INFOBUY_CKPT/intermediate" \
  "$INFOBUY_CKPT/filter" \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/raw_results" \
  "$STORAGE_PATH/rechecked" \
  "$STORAGE_PATH/summaries" \
  "$STORAGE_PATH/reports" \
  "$INFOBUY_LOGS/sft" \
  "$INFOBUY_LOGS/rl" \
  "$INFOBUY_LOGS/vllm" \
  "$INFOBUY_LOGS/eval" \
  "$INFOBUY_LOGS/wandb" \
  "$INFOBUY_SERVICES/teacher_vllm" \
  "$INFOBUY_SERVICES/student_vllm" \
  "$INFOBUY_TMP/downloads" \
  "$INFOBUY_TMP/extraction" \
  "$INFOBUY_TMP/debug" \
  "$INFOBUY_BACKUPS"

echo "Storage tree under $INFOBUY_STORE:"
find "$INFOBUY_STORE" -maxdepth 3 -type d | sort
