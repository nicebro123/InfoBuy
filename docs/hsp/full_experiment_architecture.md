# HSP Full Experiment Architecture

This document is the canonical full-data experiment plan. Smoke runs are only
runtime checks and must not be reported as experiment results.

## Data Contract

| Stage | File | Expected rows | Purpose |
| --- | --- | ---: | --- |
| Source pool | `raw/numinamath_cot_synthetic_math_seed_v0_1500.jsonl` | 1500 | Provenance pool |
| Full train split | `raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl` | 800 | RL training |
| Full validation split | `raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl` | 200 | RL validation |
| Protocol SFT train | `protocol/hsp_protocol_train_pilot_v1.jsonl` | 4800 | ASK/VERIFY/ACCEPT behavior |
| Protocol SFT validation | `protocol/hsp_protocol_validation_pilot_v1.jsonl` | 1200 | Protocol probe |
| Optional action boost | `protocol/hsp_protocol_train_pilot_v1_action_boost_x3.jsonl` | 12000 | Protocol variant only |

The SFT stage uses protocol data. The RL stage uses the raw math split with
`question` as the prompt and `gold_answer` as reward ground truth.

## Full Evaluation Contract

Full evaluation means no `MAX_EXAMPLES` limit.

| Task | Source | Expected examples |
| --- | --- | ---: |
| `math` | local MATH500 CSV | 500 |
| `gsm8k` | cached `openai/gsm8k`, test | 1319 |
| `minerva` | cached `zwhe99/simplerl-minerva-math`, test | 272 |
| `olympiad` | cached `zwhe99/simplerl-OlympiadBench`, test | 675 |
| `aime2024` | cached `HuggingFaceH4/aime_2024`, train | 30 |
| `aime2025` | cached `yentinglin/aime_2025`, train | 30 |

The default evaluation budget is `samples_per_question=8`, so one full pass
contains 22,608 generations per model and collection mode.

## Minimum Two-GPU Layout

The full stack must be launchable with two GPUs:

| Role | Default | Purpose |
| --- | --- | --- |
| Teacher | GPU 0 | Qwen3-8B service on port 7778 |
| Worker queue | GPU 1 | Student SFT/RL/eval queue |

Use more workers by setting `WORKER_GPUS`, for example:

```bash
TEACHER_GPU=0 WORKER_GPUS="1;2;3" bash scripts/launch_hsp_full_stack.sh
```

With only two GPUs:

```bash
TEACHER_GPU=0 WORKER_GPUS=1 bash scripts/launch_hsp_full_stack.sh
```

The teacher GPU must not appear in `WORKER_GPUS`.

## Required Experiment Matrix

Use `configs/experiments/hsp_official.yaml` as the paper-facing queue.

| Group | Runs | Purpose |
| --- | --- | --- |
| Baseline | SFT-only full eval | Protocol-SFT model before RL |
| Main | `qwen3_hsp_grpo_main` | Main HSP GRPO result |
| Reward shaping | `qwen3_hsp_grpo_shaped` | Shaped reward comparison |
| Cost | `no_cost`, `cost_005`, `cost_030` | Teacher-token cost sensitivity |
| Trust | `trust_025`, `trust_080` | Acceptance/reliability sensitivity |
| Budget | `budget_small`, `budget_large` | ASK/VERIFY budget sensitivity |
| Interaction depth | `interactions_1`, `interactions_4` | Number of HSP opportunities |
| Hyperparams | `kl_003`, `kl_030`, `lr_3e-7`, `lr_3e-6` | Stability/sensitivity |
| Optional protocol variant | action-boost SFT and downstream RL | Protocol data scaling |

## RelayLLM-Inspired Analysis Layer

Use `configs/experiments/hsp_analysis.yaml` for analysis runs that should not
be mixed into the main-result queue:

| Group | Runs | Purpose |
| --- | --- | --- |
| Reward components | `shaped_no_independent_bonus`, `shaped_no_exploration_reward` | Test whether the policy learns useful help-seeking or over-relies on the teacher |
| Protocol penalties | `shaped_no_protocol_penalty` | Test whether the action grammar penalties are needed |
| Trust penalties | `shaped_no_accept_penalty`, `shaped_no_reject_penalty` | Test incorrect accept/reject behavior |
| Teacher failures | `shaped_no_teacher_error_penalty` | Test robustness to teacher service failures |
| Fixed budgets | `fixed_budget_32_32`, `fixed_budget_64_96`, `fixed_budget_128_192` | Compare fixed teacher-token budgets with the default dynamic request length |

Use `scripts/run_hsp_analysis_eval.sh` for evaluation-only analyses:

| Suite | Command suffix | Purpose |
| --- | --- | --- |
| Full policy | `full` | Standard six-benchmark HSP policy evaluation |
| Teacher-free | `teacher-free` | Disable help and test whether the student retained reasoning ability |
| Forced collection | `forced` | Collect forced ASK and forced VERIFY trajectories |
| Fixed budget | `fixed-budget` | Evaluate fixed small/mid/large ASK/VERIFY budgets |
| OOD | `ood` | Evaluate BBEH, MMLU-Pro, and SuperGPQA generalization |

## Execution

Prepare the full SpecFlow-style two-GPU queue:

```bash
bash scripts/run_full_pipeline_2gpu.sh
bash scripts/run_full_pipeline_2gpu.sh --launch
```

Preflight:

```bash
source setup/env.sh
python scripts/check_hsp_full_data.py --strict
```

Train protocol SFT once:

```bash
bash run.sh sft --gpu 1
bash run.sh token-probe --gpu 1
```

Launch the full RL queue:

```bash
TEACHER_GPU=0 WORKER_GPUS=1 bash scripts/launch_hsp_full_stack.sh --skip-smoke
```

Launch the analysis RL queue:

```bash
TEACHER_GPU=0 WORKER_GPUS=1 SPEC=configs/experiments/hsp_analysis.yaml \
bash scripts/launch_hsp_full_stack.sh --skip-smoke
```

Evaluate full tasks:

```bash
bash run.sh teacher --gpu 0 --port 7778
MODEL_PATH="$INFOBUY_CKPT/sft/qwen3-0.6b-hsp-sft" bash run.sh eval --gpu 1 --port 7778
MODEL_PATH="/path/to/merged/rl/checkpoint" bash run.sh eval --gpu 1 --port 7778
MODEL_PATH="/path/to/merged/rl/checkpoint" GPU_QUEUE=1 bash scripts/run_hsp_analysis_eval.sh all
```

Do not set `MAX_EXAMPLES` for paper results.

## Acceptance Gates

- `python scripts/check_hsp_full_data.py --strict` passes.
- `bash run.sh checks` passes.
- `run.sh eval` honors `--model-path` or `MODEL_PATH`.
- Full eval runs with no `MAX_EXAMPLES`.
- Every reported result has `run_config.yaml`, `train.log`, checkpoint path,
  eval JSON, and recheck summary when recheck is enabled.
