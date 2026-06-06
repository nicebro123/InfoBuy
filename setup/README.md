# InfoBuy Storage Setup

This directory contains storage/bootstrap scripts. It does not install Python
packages and it does not download anything unless you explicitly run the
download scripts.

The rule is strict:

```text
InfoBuy/           code, configs, scripts, small docs
InfoBuy_store/     models, HF cache, datasets, checkpoints, eval outputs
```

The code repository should stay lightweight. Large files should not be placed
inside `InfoBuy/`.

## Directory Layout

```text
$INFOBUY_STORE/
├── huggingface/
│   ├── cache/                    # HF_HOME
│   ├── hub/                      # HF_HUB_CACHE
│   └── datasets_cache/           # HF_DATASETS_CACHE
├── models/
│   ├── pretrained/
│   │   ├── Qwen3-0.6B/           # student base
│   │   └── Qwen3-8B/             # main teacher
│   └── teachers/
│       └── qwen3-8b-main -> ../pretrained/Qwen3-8B
├── datasets/
│   ├── hf_downloads/
│   │   ├── AI-MO__NuminaMath-CoT/
│   │   └── optional_baselines/
│   │       ├── dapo17k_raw/              # optional
│   │       └── dapo17k_8b_filtered/      # optional
│   ├── benchmarks/
│   │   ├── math500/ gsm8k/ aime2024/ aime2025/
│   │   ├── amc23/ minerva/ olympiadbench/
│   └── infobuy/
│       ├── raw/                  # decontaminated NuminaMath source splits
│       ├── protocol/             # Ask/Verify/Accept SFT protocol data
│       ├── flat/                 # flattened SFT data
│       ├── replay/               # outcome-selected rollout replay data
│       ├── trust/                # trust/acceptance data
│       ├── purchase/             # information-purchase data
│       ├── manifests/            # provenance and split manifests
│       └── splits/               # derived split metadata
├── checkpoints/
│   ├── sft/
│   ├── rl/
│   ├── merged/
│   ├── intermediate/
│   └── filter/
├── eval/
│   ├── evaluation/               # eval/generate_withhelp.py raw run outputs
│   ├── raw_results/
│   ├── rechecked/
│   ├── summaries/
│   └── reports/
├── logs/
│   ├── sft/ rl/ vllm/ eval/ wandb/
├── services/
│   ├── teacher_vllm/ student_vllm/
├── backups/                    # moved local data backups, never code
└── tmp/
    ├── downloads/ extraction/ debug/
```

## Quick Start

```bash
# 1. Choose your external storage root.
export INFOBUY_STORE=/Users/quanquan/Desktop/InfoBuy_store

# 2. Load path variables.
source setup/env.sh

# 3. Create the storage tree.
bash setup/make_dirs.sh

# 4. Optional legacy bridge only:
#    create InfoBuy/data -> $INFOBUY_STORE/datasets for old commands.
# bash setup/link_data.sh

# 5. Download model weights and dataset/cache assets when ready.
# bash setup/download_models.sh
# bash setup/download_data.sh
```

## Path Variables

`setup/env.sh` defines the new `INFOBUY_*` variables and also exports old
`HSP_*` aliases for backward compatibility.

| Use | Path |
|:--|:--|
| External store root | `$INFOBUY_STORE` |
| Student base model | `$INFOBUY_PRETRAINED_MODELS/Qwen3-0.6B` |
| Main teacher model | `$INFOBUY_TEACHER_MODELS/qwen3-8b-main` |
| HF dataset/model snapshots | `$INFOBUY_HF_DOWNLOADS` |
| Benchmarks | `$INFOBUY_BENCHMARKS` |
| InfoBuy generated data | `$INFOBUY_GENERATED_DATA` |
| SFT checkpoints | `$INFOBUY_CKPT/sft/<run_name>` |
| GRPO checkpoints | `$INFOBUY_CKPT/rl/<run_name>` |
| Merged HF-format models | `$INFOBUY_CKPT/merged/<run_name>` |
| Evaluation outputs | `$STORAGE_PATH` |
| Local backups moved out of repo | `$INFOBUY_BACKUPS` |

## Data Policy

The current main line uses decontaminated NuminaMath-CoT `synthetic_math`
splits:

```text
$INFOBUY_GENERATED_DATA/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl
$INFOBUY_GENERATED_DATA/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl
```

Protocol SFT data lives under:

```text
$INFOBUY_GENERATED_DATA/protocol/
```

Outcome replay, trust data, and information-purchase data live under:

```text
$INFOBUY_GENERATED_DATA/replay/
$INFOBUY_GENERATED_DATA/trust/
$INFOBUY_GENERATED_DATA/purchase/
```

DAPO is not the default InfoBuy training data. It is optional for RelayLLM
baseline or ablation experiments:

```bash
DOWNLOAD_DAPO_BASELINES=1 bash setup/download_data.sh
```

Those optional assets are stored under:

```text
$INFOBUY_HF_DOWNLOADS/optional_baselines/
```

## Notes

- Models are downloaded with `--local-dir` because scripts load them by path.
- HF-id datasets used by `load_dataset("...")` are cached under `HF_HOME`.
- `setup/link_data.sh` is optional and exists only for older commands that still
  expect `data/...` paths inside the repo.
- Do not commit files from `$INFOBUY_STORE`.
