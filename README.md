# InfoBuy

InfoBuy studies small-large model collaborative reasoning as an information
purchase problem: the small model learns when to buy a hint, when to buy
verification, how many teacher tokens to buy, and whether to trust the purchased
information.

The implementation still uses the HSP protocol internally:

```text
<ASK>N</ASK>       buy bounded reasoning help
<VERIFY>N</VERIFY> buy bounded verification
<ACCEPT>           explicitly trust/adopt verified feedback
```

## Repository Layout

```text
InfoBuy/
├── SFT_stage/          protocol SFT data builders, collator, trainer, preflight
├── RL_stage/           GRPO config, HSP rollout state machine, reward function
├── eval/               collaborative generation and benchmark evaluation
├── setup/              external storage, download, and environment scripts
├── docs/hsp/           detailed method, data, reward, and training docs
├── utils/              teacher service utilities
└── README_HSP.md       detailed end-to-end technical manual
```

Large files do not belong in this repository.

## External Store Layout

Use one external directory for models, datasets, checkpoints, logs, and
evaluation outputs:

```bash
# Optional: choose a large external disk. If unset, setup/env.sh defaults to
# a sibling directory next to the repo, such as ../InfoBuy_store.
export INFOBUY_STORE=$HOME/InfoBuy_store
source setup/env.sh
bash setup/make_dirs.sh
```

The store is organized as:

```text
$INFOBUY_STORE/
├── huggingface/        HF cache, hub cache, dataset cache
├── models/             pretrained student/teacher weights and aliases
├── datasets/           HF snapshots, benchmarks, generated InfoBuy data
├── checkpoints/        SFT, RL, merged, intermediate checkpoints
├── eval/               generation outputs, rechecked results, summaries
├── logs/               SFT/RL/vLLM/eval/W&B logs
├── services/           teacher/student service runtime files
├── backups/            historical local data backups moved out of the repo
└── tmp/                downloads, extraction, debug scratch
```

Detailed storage rules are in
[`docs/hsp/storage_layout.md`](docs/hsp/storage_layout.md).

## Main Data Location

The main generated data lives outside the repo:

```text
$INFOBUY_GENERATED_DATA = $INFOBUY_STORE/datasets/infobuy
```

Current main training files:

```text
$INFOBUY_GENERATED_DATA/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl
$INFOBUY_GENERATED_DATA/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl
$INFOBUY_GENERATED_DATA/protocol/hsp_protocol_train_pilot_v1.jsonl
$INFOBUY_GENERATED_DATA/protocol/hsp_protocol_validation_pilot_v1.jsonl
```

`setup/link_data.sh` is only a legacy bridge for older commands that require
`data/...` paths. The preferred setup does not require a `data` directory inside
the code repository.
