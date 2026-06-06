#!/usr/bin/env bash
# Download base/teacher model weights to explicit folders under
# $INFOBUY_MODELS/pretrained.
# Models are loaded BY PATH (vllm_service --model_path, train_hsp --model_name),
# so an explicit --local-dir is the right choice here.
#
# Usage:  source setup/env.sh && bash setup/download_models.sh
set -euo pipefail
: "${INFOBUY_PRETRAINED_MODELS:?source setup/env.sh first}"
: "${INFOBUY_TEACHER_MODELS:?source setup/env.sh first}"

# Use `hf download` on newer huggingface_hub; falls back to huggingface-cli.
DL="huggingface-cli download"
command -v hf >/dev/null 2>&1 && DL="hf download"

# Student base (SFT cold-start)
$DL Qwen/Qwen3-0.6B --local-dir "$INFOBUY_PRETRAINED_MODELS/Qwen3-0.6B"

# Teacher (vllm_service teacher + difficulty calibration). ~16GB.
$DL Qwen/Qwen3-8B --local-dir "$INFOBUY_PRETRAINED_MODELS/Qwen3-8B"

ln -sfn "$INFOBUY_PRETRAINED_MODELS/Qwen3-8B" "$INFOBUY_TEACHER_MODELS/qwen3-8b-main"

# Optional: stronger teacher for hard RL data (uncomment if needed). ~28GB+.
if [ "${DOWNLOAD_OPTIONAL_TEACHERS:-0}" = "1" ]; then
  $DL Qwen/Qwen3-14B-Base --local-dir "$INFOBUY_PRETRAINED_MODELS/Qwen3-14B-Base"
  ln -sfn "$INFOBUY_PRETRAINED_MODELS/Qwen3-14B-Base" "$INFOBUY_TEACHER_MODELS/qwen3-14b-strong"
fi

echo "Pretrained models in $INFOBUY_PRETRAINED_MODELS:"
ls -1 "$INFOBUY_PRETRAINED_MODELS"
echo "Teacher aliases in $INFOBUY_TEACHER_MODELS:"
ls -l "$INFOBUY_TEACHER_MODELS"
