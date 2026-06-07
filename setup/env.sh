#!/usr/bin/env bash
# InfoBuy storage environment. Usage: source setup/env.sh
#
# Keep the code repository small. All model weights, Hugging Face caches,
# downloaded datasets, generated datasets, checkpoints, logs, and evaluation
# outputs live under $INFOBUY_STORE outside the repo.

# ---- CHANGE THIS to your large disk / mount point if needed ----
# Default to a sibling directory next to the code repo, e.g.:
#   /path/to/InfoBuy       -> /path/to/InfoBuy_store
#   /path/to/InfoBuy-main  -> /path/to/InfoBuy-main_store
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _INFOBUY_ENV_FILE="${BASH_SOURCE[0]}"
else
  _INFOBUY_ENV_FILE="$0"
fi

_INFOBUY_ENV_DIR="$(cd "$(dirname "${_INFOBUY_ENV_FILE}")" && pwd)"
_INFOBUY_REPO_ROOT="$(cd "${_INFOBUY_ENV_DIR}/.." && pwd)"
_INFOBUY_DEFAULT_STORE="$(dirname "${_INFOBUY_REPO_ROOT}")/$(basename "${_INFOBUY_REPO_ROOT}")_store"
export INFOBUY_STORE="${INFOBUY_STORE:-${HSP_STORE:-${_INFOBUY_DEFAULT_STORE}}}"

# Backward-compatible alias used by older HSP scripts/docs.
export HSP_STORE="$INFOBUY_STORE"

# Hugging Face cache goes on the external store.
export HF_HOME="${HF_HOME:-$INFOBUY_STORE/huggingface/cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$INFOBUY_STORE/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$INFOBUY_STORE/huggingface/datasets_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1          # faster parallel downloads
# export HF_HUB_OFFLINE=1                    # uncomment after downloads to force offline

# eval/generate_withhelp.py writes results under $STORAGE_PATH/evaluation/...
export STORAGE_PATH="${STORAGE_PATH:-$INFOBUY_STORE/eval}"

# Convenience handles used by the other setup scripts.
export INFOBUY_MODELS="$INFOBUY_STORE/models"
export INFOBUY_DATASETS="$INFOBUY_STORE/datasets"
export INFOBUY_CKPT="$INFOBUY_STORE/checkpoints"
export INFOBUY_LOGS="$INFOBUY_STORE/logs"
export INFOBUY_SERVICES="$INFOBUY_STORE/services"
export INFOBUY_TMP="$INFOBUY_STORE/tmp"
export INFOBUY_BACKUPS="$INFOBUY_STORE/backups"

export INFOBUY_PRETRAINED_MODELS="$INFOBUY_MODELS/pretrained"
export INFOBUY_TEACHER_MODELS="$INFOBUY_MODELS/teachers"
export INFOBUY_HF_DOWNLOADS="$INFOBUY_DATASETS/hf_downloads"
export INFOBUY_BENCHMARKS="$INFOBUY_DATASETS/benchmarks"
export INFOBUY_GENERATED_DATA="$INFOBUY_DATASETS/infobuy"

# Backward-compatible HSP aliases. New scripts should prefer INFOBUY_*.
export HSP_MODELS="$INFOBUY_MODELS"
export HSP_DATASETS="$INFOBUY_DATASETS"
export HSP_CKPT="$INFOBUY_CKPT"
export HSP_LOGS="$INFOBUY_LOGS"
export HSP_SERVICES="$INFOBUY_SERVICES"
export HSP_TMP="$INFOBUY_TMP"
export HSP_BACKUPS="$INFOBUY_BACKUPS"
export HSP_PRETRAINED_MODELS="$INFOBUY_PRETRAINED_MODELS"
export HSP_TEACHER_MODELS="$INFOBUY_TEACHER_MODELS"
export HSP_HF_DOWNLOADS="$INFOBUY_HF_DOWNLOADS"
export HSP_BENCHMARKS="$INFOBUY_BENCHMARKS"
export HSP_GENERATED_DATA="$INFOBUY_GENERATED_DATA"

echo "INFOBUY_STORE       = $INFOBUY_STORE"
echo "HF_HOME             = $HF_HOME"
echo "HF_HUB_CACHE        = $HF_HUB_CACHE"
echo "HF_DATASETS_CACHE= $HF_DATASETS_CACHE"
echo "STORAGE_PATH        = $STORAGE_PATH"
echo "models              = $INFOBUY_MODELS"
echo "datasets            = $INFOBUY_DATASETS"
echo "checkpoints         = $INFOBUY_CKPT"
echo "generated_data      = $INFOBUY_GENERATED_DATA"
echo "backups             = $INFOBUY_BACKUPS"
