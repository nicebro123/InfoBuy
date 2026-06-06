#!/usr/bin/env bash
# Download datasets. Two cases are handled differently on purpose:
#
#   (A) Immutable HF dataset snapshots kept for audit/reference -> explicit
#       --local-dir under $INFOBUY_DATASETS/hf_downloads.
#
#   (B) Datasets the code loads BY HF id at runtime -> into the HF CACHE (no --local-dir)
#       e.g. eval/datasets_loader.py calls load_dataset("openai/gsm8k") etc.
#       These MUST be in HF_HOME or load_dataset(id) won't find them. The cache is
#       already on the external store, so they're still centralized.
#
# Usage:  source setup/env.sh && bash setup/download_data.sh
set -euo pipefail
: "${INFOBUY_HF_DOWNLOADS:?source setup/env.sh first}"
: "${INFOBUY_BENCHMARKS:?source setup/env.sh first}"

DL="huggingface-cli download"
command -v hf >/dev/null 2>&1 && DL="hf download"

# ---------------------------------------------------------------------------
# (A) Explicit local snapshots
# ---------------------------------------------------------------------------
# Main InfoBuy source pool. The pipeline builds decontaminated JSONL files under
# $INFOBUY_GENERATED_DATA via SFT_stage/fetch_hsp_source_dataset.py; this
# immutable copy is for offline audit/reference.
$DL AI-MO/NuminaMath-CoT --repo-type dataset --local-dir "$INFOBUY_HF_DOWNLOADS/AI-MO__NuminaMath-CoT"

# DAPO is not part of the HSP main line. Download it only for RelayLLM
# baseline/ablation experiments.
if [ "${DOWNLOAD_DAPO_BASELINES:-0}" = "1" ]; then
  $DL ChengsongHuang/8B_filtered_data --repo-type dataset \
    --local-dir "$INFOBUY_HF_DOWNLOADS/optional_baselines/dapo17k_8b_filtered"
  $DL guanning-ai/dapo17k --repo-type dataset \
    --local-dir "$INFOBUY_HF_DOWNLOADS/optional_baselines/dapo17k_raw"
fi

# ---------------------------------------------------------------------------
# (B) HF-id runtime datasets -> into the HF cache (NO --local-dir)
#     Exact ids from eval/datasets_loader.py
# ---------------------------------------------------------------------------
for ID in \
  openai/gsm8k \
  zwhe99/amc23 \
  zwhe99/simplerl-minerva-math \
  zwhe99/simplerl-OlympiadBench \
  HuggingFaceH4/aime_2024 \
  yentinglin/aime_2025 ; do
    echo ">> caching $ID"
    $DL "$ID" --repo-type dataset
done

# MATH-500 is a CSV URL (NOT a HF dataset) pulled at runtime by MathDatasetHandler.
if [ "${DOWNLOAD_MATH500_CSV:-1}" = "1" ]; then
  curl -L -o "$INFOBUY_BENCHMARKS/math500/math_500_test.csv" \
    https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv
fi

echo "Done."
echo "Explicit HF snapshots: $INFOBUY_HF_DOWNLOADS"
echo "Benchmark files:       $INFOBUY_BENCHMARKS"
echo "HF-id datasets cached in HF_HOME=$HF_HOME"
