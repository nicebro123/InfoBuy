# Help-Seeking Policy (HSP): Teaching Small Models When to Ask, Verify, and Trust

> Train a small language model to act as an **active collaboration controller** -- learning *when* to ask for help, *when* to request verification, and *when* to trust external feedback during step-by-step reasoning.

> **This README is the single entry point for the HSP project.** Run the full data
> pipeline with `python build_and_upload_hsp_dataset.py --build_only`. Detailed
> design specs live under [`docs/hsp/`](docs/hsp/). This project builds on the
> RelayLLM collaborative-decoding framework (the `verl` engine under `RL_stage/`).

### Documentation Map

| Document | Purpose |
|:---------|:--------|
| **README_HSP.md** (this file) | End-to-end pipeline, commands, project layout |
| [`docs/hsp/design.md`](docs/hsp/design.md) | Full research design |
| [`docs/hsp/dataset_pipeline.md`](docs/hsp/dataset_pipeline.md) | Dataset construction spec |
| [`docs/hsp/sft_samples.md`](docs/hsp/sft_samples.md) | The 6 SFT sample types with examples |
| [`docs/hsp/reward_design.md`](docs/hsp/reward_design.md) | Outcome-cost-trust reward spec |
| [`docs/hsp/rl_training_strategy.md`](docs/hsp/rl_training_strategy.md) | GRPO training strategy |
| [`docs/hsp/training_testing.md`](docs/hsp/training_testing.md) | Operational run/test checklist |
| [`docs/hsp/storage_layout.md`](docs/hsp/storage_layout.md) | Storage/folder plan: weights, datasets, cache, checkpoints (all outside the repo) |
| [`docs/legacy/`](docs/legacy/) | Baseline & upstream reference docs (comparison only) |

---

## Table of Contents

- [Research Motivation](#research-motivation)
- [Method Overview](#method-overview)
  - [Three-Token Protocol](#three-token-protocol)
  - [Training Pipeline](#training-pipeline)
- [Environment Setup](#environment-setup)
- [Complete Pipeline](#complete-pipeline)
  - [Step 1: Source Data Preparation](#step-1-source-data-preparation)
  - [Step 2: Protocol SFT Data Construction](#step-2-protocol-sft-data-construction)
  - [Step 3: SFT Training (Protocol Warm-Up)](#step-3-sft-training-protocol-warm-up)
  - [Step 4: Start Teacher Service](#step-4-start-teacher-service)
  - [Step 5: Evaluation (Pre-RL Baseline)](#step-5-evaluation-pre-rl-baseline)
  - [Step 6: Outcome Replay SFT (Optional)](#step-6-outcome-replay-sft-optional)
  - [Step 7: RL Training (GRPO)](#step-7-rl-training-grpo)
  - [Step 8: Model Merging](#step-8-model-merging)
  - [Step 9: Final Evaluation](#step-9-final-evaluation)
- [Project Structure](#project-structure)
- [Reward Function Design](#reward-function-design)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Research Motivation

Existing small-large model collaboration approaches operate at **query-level** granularity: a router decides whether to send the entire question to a small or large model. This is wasteful -- the small model could handle most reasoning steps independently and only needs help at critical moments.

HSP takes a fundamentally different approach. Instead of passively being routed, the small model learns to **actively manage collaboration** during the reasoning process. It makes three distinct types of decisions:

1. **When to ask for help** -- capability-level: "I'm stuck and need external reasoning support."
2. **When to request verification** -- cognition-level: "I have an answer but I'm unsure if it's correct."
3. **When to trust feedback** -- belief-level: "I've received a review; should I update my answer?"

This transforms the small model from a passive routing target into an **active controller** with metacognitive collaboration abilities.

---

## Method Overview

### Three-Token Protocol

HSP introduces three special action tokens that the student model learns to generate:

```
Token              Level         Semantics
-----------        ----------    --------------------------------------------------------
<ASK>N</ASK>       Capability    "I can't proceed; give me a hint (max N tokens)."
<VERIFY>N</VERIFY> Cognition     "I have a tentative answer; review it (max N tokens)."
<ACCEPT>           Belief        "I've seen the feedback and I choose to adopt it."
```

The integer N controls the teacher's token budget, analogous to RelayLLM's `<call>N</call>`.
Typical values: ASK ∈ {32, 64, 96, 128}, VERIFY ∈ {64, 96, 128}.

**Interaction Example:**

```text
Student: We need to solve ... but I'm unsure how to simplify the next step.
         <ASK>64</ASK>

Teacher: <TEACHER_HELP>
         Use substitution u = x^2 to simplify the integral.
         </TEACHER_HELP>

Student: Following that hint, we substitute u = x^2, so du = 2x dx ...
         The final answer is \boxed{42}.
```

```text
Student: I think the answer is \boxed{42}.
         <VERIFY>96</VERIFY>

Teacher: <TEACHER_REVIEW>
         Verdict: incorrect
         Issue: The multiplication in step 2 is wrong: 7 * 5 = 35, not 42.
         Correction: \boxed{35}
         </TEACHER_REVIEW>

Student: <ACCEPT>
         After correcting the calculation, the answer is \boxed{35}.
```

Key design properties:

- `<ASK>N</ASK>` **implicitly accepts** the teacher's help (used as reasoning context). N controls the teacher response budget.
- `<VERIFY>N</VERIFY>` does **not** automatically accept feedback. The student must explicitly output `<ACCEPT>` to adopt corrections.
- If the student does not output `<ACCEPT>` after a review, it may retain its original answer -- this models **selective trust**.
- The token budget N lets the student learn cost-aware help-seeking: requesting fewer tokens when a short hint suffices, more when detailed verification is needed.

### Training Pipeline

```
Step 1: Source Data Preparation
  fetch_hsp_source_dataset.py   Fetch NuminaMath-CoT synthetic_math problems
  build_hsp_source_splits.py    Decontaminate and split into train/val
          |
          v
Step 2: Protocol SFT Data Construction
  build_hsp_sft.py              Generate 6 sample types:
                                  - normal (no interaction)
                                  - ask_help
                                  - verify_confirm
                                  - verify_accept_correction
                                  - verify_reject_bad_feedback
                                  - verify_uncertain
          |
          v
Step 3: SFT Training
  train_hsp.py                  Segment-aware training with student-only loss
  hsp_collator.py               Offset-based loss masking (teacher tokens excluded)
          |
          v
Step 4: Teacher Service
  vllm_service.py               Flask HTTP service wrapping vLLM
          |
          v
Step 5: Pre-RL Evaluation
  generate_withhelp.py           HSP collaborative generation
  evaluate_forhelp.bash          Batch evaluation on 6 benchmarks
          |
          v
Step 6: Outcome Replay SFT (Optional)
  collect_hsp_candidates.bash    Collect counterfactual rollouts
  build_hsp_outcome_sft.py       Select best trajectories per problem
  mix_hsp_sft.py                 Mix protocol + replay data
          |
          v
Step 7: RL Training (GRPO)
  qwen3_hsp_grpo.sh              Launcher (runs preflight before training)
  config_hsp.yaml                GRPO config with HSP interaction policy
  math_hsp_group.py              Outcome-cost-trust reward function
  help_vllm_rollout_spmd.py      HSP state machine rollout engine
          |
          v
Step 8: Model Merging
  model_merger.py                Merge FSDP shards into HuggingFace format
          |
          v
Step 9: Final Evaluation
  evaluate_forhelp.bash           Batch evaluation with interaction_policy=hsp
  summarize_hsp_results.py        Action calibration analysis
```

---

## Environment Setup

```bash
# Clone repository
git clone <repo_url>
cd InfoBuy

# Install dependencies
pip install -r requirements.txt

# Configure external storage for models, HF cache, generated data, checkpoints,
# and evaluation outputs.
export INFOBUY_STORE=/Users/quanquan/Desktop/InfoBuy_store
source setup/env.sh
bash setup/make_dirs.sh

# Optional legacy bridge only: create InfoBuy/data -> $INFOBUY_STORE/datasets.
# bash setup/link_data.sh

# Optional downloads; these are large.
# bash setup/download_models.sh
# bash setup/download_data.sh

# Other environment variables
export WANDB_API_KEY=your_key               # optional, for W&B logging
export OPENAI_API_KEY=your_key              # optional, for answer recheck
export SKIP_LLM_RECHECK=1                  # set this if no OpenAI key
```

**Hardware requirements:**
- SFT training: 1 GPU (24 GB+ VRAM recommended for Qwen3-0.6B)
- Teacher service: 1 GPU (Qwen3-8B requires ~20 GB VRAM)
- RL training (GRPO): 1 GPU for actor/ref + teacher running on a separate GPU
- Evaluation: 1 GPU for student vLLM + teacher service on another GPU

---

## Complete Pipeline

### Step 1: Source Data Preparation

Fetch the NuminaMath-CoT `synthetic_math` source pool from HuggingFace. This category is chosen to avoid contamination with held-out benchmarks (MATH-500, GSM8K, AMC, AIME, Olympiad, Minerva).

```bash
# Fetch 1000 problems from NuminaMath-CoT/synthetic_math
python -m SFT_stage.fetch_hsp_source_dataset \
    --max_records 1000 \
    --request_interval_seconds 2 \
    --output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl \
    --manifest ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_seed_v0_1000.manifest.json
```

Split into train/validation **before** protocol expansion (prevents same-problem leakage). The splitter automatically fetches held-out benchmark questions and removes exact/near-duplicate matches (Jaccard threshold = 0.85):

```bash
python -m SFT_stage.build_hsp_source_splits \
    --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl \
    --train_size 800 \
    --validation_size 200 \
    --seed 42 \
    --train_output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
    --validation_output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl \
    --manifest ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_split_pilot_v1.manifest.json
```

**Output:**
- `${INFOBUY_GENERATED_DATA}/raw/*_train_pilot_v1_800.jsonl` -- 800 decontaminated train problems
- `${INFOBUY_GENERATED_DATA}/raw/*_validation_pilot_v1_200.jsonl` -- 200 decontaminated validation problems
- `${INFOBUY_GENERATED_DATA}/manifests/*.manifest.json` -- provenance and decontamination audit trail

### Step 1b: Difficulty Calibration (optional, needs GPU)

**The cold-start SFT dataset teaches the protocol tokens (`<ASK>N</ASK>`,
`<VERIFY>N</VERIFY>`, `<ACCEPT>`), so it should contain BOTH easy and hard problems** --
the model must learn the tokens across the full difficulty range. Do *not* narrow the
SFT pool to a band.

This step is therefore used in **`tag` mode** for SFT: run a strong reference model
(e.g. **Qwen3-8B**) over each problem, annotate `solve_rate`, and report the difficulty
histogram so you can confirm the spread covers easy → hard. It keeps every problem:

```bash
python -m SFT_stage.calibrate_hsp_difficulty \
    --input  ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
    --output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_calibrated_800.jsonl \
    --report ${INFOBUY_GENERATED_DATA}/manifests/difficulty_train.json \
    --model_path Qwen/Qwen3-8B --samples_k 4 --mode tag
```

The JSON report shows the aggregate accuracy (mean `solve_rate`) and the difficulty
histogram. Modes:

| `--mode` | Keeps | Use for |
|:---------|:------|:--------|
| `tag` (default) | **all** problems, annotated | SFT protocol learning (easy + hard) |
| `stratify` | balanced easy/medium/hard mix | SFT, to guarantee balanced coverage |
| `target` (`--target_accuracy 0.70`) | subset centered on ~70% | **RL stage** help-seeking band |
| `band` (`--min/--max_solve_rate`) | explicit solve-rate band | custom selection |

> One-command pipeline (annotates difficulty, keeps the full spread):
> `python build_and_upload_hsp_dataset.py --build_only --calibrate`
> (the `~70%` band is an RL-stage concern: add `--calibration_mode target` only there;
> omit `--calibrate` entirely to skip on a machine without a GPU.)

### Step 2: Protocol SFT Data Construction

Generate structured SFT data with 6 sample types. Each example contains `segments` with explicit `source` (student/teacher/user) and `loss` flags:

```bash
# Training data: weighted random sampling (1 variant per problem)
python -m SFT_stage.build_hsp_sft \
    --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
    --output ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
    --variants_per_problem 1 \
    --seed 42

# Validation data: all 6 types per problem (for coverage)
python -m SFT_stage.build_hsp_sft \
    --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl \
    --output ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_validation_pilot_v1.jsonl \
    --emit_all_types \
    --seed 42
```

**6 Sample Types and Their Weights:**

| Type | Weight | Purpose |
|:-----|:------:|:--------|
| `normal` | 45% | No interaction -- prevents "always ask" collapse |
| `ask_help` | 15% | Student stuck, outputs `<ASK>N</ASK>`, receives help, continues |
| `verify_confirm` | 10% | Student correct, `<VERIFY>N</VERIFY>` confirms, no `<ACCEPT>` needed |
| `verify_accept_correction` | 15% | Student wrong, `<VERIFY>N</VERIFY>` finds error, `<ACCEPT>` + correct |
| `verify_reject_bad_feedback` | 10% | Teacher feedback wrong, student rejects (no `<ACCEPT>`) |
| `verify_uncertain` | 5% | Teacher uncertain, student proceeds independently |

**Validate protocol integrity:**

```bash
python -m SFT_stage.preflight_hsp \
    --dataset ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
    --require_all_types \
    --require_context_tokens
```

### Step 3: SFT Training (Protocol Warm-Up)

Train the student model with **segment-level loss masking**. The `HSPDataCollator` uses `offset_mapping` from the fast tokenizer to precisely mask teacher tokens from the loss -- only student-generated tokens contribute to gradient updates.

```bash
python -m SFT_stage.train_hsp \
    --model_name Qwen/Qwen3-0.6B \
    --dataset ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
    --output_dir ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
    --max_seq_length 12288 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16
```

The trainer automatically:
- Adds `<ASK>`, `</ASK>`, `<VERIFY>`, `</VERIFY>`, `<ACCEPT>` as single special tokens
- Adds `<TEACHER_HELP>`, `</TEACHER_HELP>`, `<TEACHER_REVIEW>`, `</TEACHER_REVIEW>`, `<ENVIRONMENT_NOTICE>`, `</ENVIRONMENT_NOTICE>` as context markers
- Resizes the embedding layer accordingly
- Saves `hsp_training_contract.json` for RL compatibility verification
- Filters out non-trainable examples (those truncated mid-interaction)

**Post-SFT verification:**

```bash
# Check that all special tokens are atomic and RL config is compatible
python -m SFT_stage.preflight_hsp \
    --model_path ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
    --rl_config RL_stage/examples/config_hsp.yaml \
    --require_context_tokens
```

### Step 4: Start Teacher Service

Launch the teacher model (e.g. Qwen3-8B) as a vLLM HTTP service:

```bash
CUDA_VISIBLE_DEVICES=1 python utils/vllm_service.py \
    --model_path ${INFOBUY_TEACHER_MODELS}/qwen3-8b-main \
    --port 7778 \
    --tensor_parallel_size 1 \
    --trust_remote_code
```

**Verify the service is running:**

```bash
curl -s http://127.0.0.1:7778/generate \
    -H "Content-Type: application/json" \
    -d '[{"prompt": "What is 2+2?", "max_tokens": 32}]'
```

### Step 5: Evaluation (Pre-RL Baseline)

Run HSP evaluation across 6 math benchmarks:

```bash
# Single dataset evaluation
python -m eval.generate_withhelp \
    --small_model ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
    --dataset math \
    --larger_model Qwen3-8B \
    --large_model_url http://127.0.0.1:7778/generate \
    --interaction_policy hsp \
    --samples_per_question 8

# Batch evaluation on all 6 datasets (math, gsm8k, minerva, olympiad, aime2024, aime2025)
bash eval/evaluate_forhelp.bash \
    ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
    Qwen3-8B \
    7778 \
    "0" \
    hsp \
    8
```

**Analyze HSP action distribution:**

```bash
python -m eval.summarize_hsp_results \
    ${STORAGE_PATH}/evaluation/*/results_math_*_hsp_rechecked.json \
    --output ${STORAGE_PATH}/summaries/hsp_sft_baseline_summary.json
```

### Step 6: Outcome Replay SFT (Optional)

This optional step improves the SFT warm-up by collecting actual student rollouts, selecting the best trajectories, and mixing them with protocol data.

**6a. Collect counterfactual rollouts from training data:**

```bash
bash eval/collect_hsp_candidates.bash \
    ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
    local_json \
    Qwen3-8B \
    7778 \
    8 \
    --name ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
    --output_tag pilot_r1 \
    --max_interactions 3 \
    --ask_budget_tokens 64 \
    --verify_budget_tokens 96
```

This runs 4 collection modes per problem: `independent`, `force_ask_first`, `force_verify_after_draft`, `policy`.

**6b. Select successful, low-cost, valid trajectories:**

```bash
python -m SFT_stage.build_hsp_outcome_sft \
    --input ${STORAGE_PATH}/evaluation/*/results_local_json_*_pilot_r1_hsp*.json \
    --output ${INFOBUY_GENERATED_DATA}/replay/hsp_outcome_replay_pilot_r1.jsonl \
    --teacher_token_budget 192 \
    --teacher_cost_weight 0.15 \
    --min_score 1.0
```

**6c. Mix protocol + replay data:**

```bash
python -m SFT_stage.mix_hsp_sft \
    --protocol_data ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
    --replay_data ${INFOBUY_GENERATED_DATA}/replay/hsp_outcome_replay_pilot_r1.jsonl \
    --output ${INFOBUY_GENERATED_DATA}/replay/hsp_sft_mixed_pilot_r1.jsonl \
    --max_replay_fraction 0.50
```

Then retrain SFT with the mixed dataset before proceeding to RL.

### Step 7: RL Training (GRPO)

Run Group Relative Policy Optimization with the HSP reward function. The launcher automatically runs `preflight_hsp.py` to verify all tokens, configs, and length contracts before training starts.

**Make sure the teacher service from Step 4 is still running on port 7778.**

```bash
cd RL_stage

MODEL_PATH=${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
SAVE_PATH=${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_main \
bash examples/qwen3_hsp_grpo.sh
```

For the shaped ablation experiment:

```bash
HSP_CONFIG=examples/config_hsp_shaped.yaml \
MODEL_PATH=${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
SAVE_PATH=${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_shaped \
bash examples/qwen3_hsp_grpo.sh
```

**Key GRPO hyperparameters** (from `config_hsp.yaml`):

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| `interaction_policy` | `hsp` | Enables HSP state machine in rollout |
| `n` (rollouts per prompt) | 8 | Group size for GRPO advantage estimation |
| `max_interactions` | 3 | Max ASK/VERIFY calls per rollout |
| `ask_budget_tokens` | 64 | Teacher help token budget per ASK |
| `verify_budget_tokens` | 96 | Teacher review token budget per VERIFY |
| `kl_coef` | 0.01 | KL penalty coefficient |
| `learning_rate` | 1e-6 | Actor learning rate |
| `global_batch_size` | 32 | Effective batch size |

### Step 8: Model Merging

RL checkpoints are saved as FSDP shards. Merge them into standard HuggingFace format before evaluation:

```bash
python RL_stage/scripts/model_merger.py \
    --local_dir ${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_main/actor
```

Output is saved to `${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_main/actor/huggingface/`.

### Step 9: Final Evaluation

Evaluate the RL-trained model:

```bash
MERGED_MODEL=${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_main/actor/huggingface

bash eval/evaluate_forhelp.bash \
    ${MERGED_MODEL} \
    Qwen3-8B \
    7778 \
    "0" \
    hsp \
    8

# Detailed action calibration analysis
python -m eval.summarize_hsp_results \
    ${STORAGE_PATH}/evaluation/*/results_math_*_hsp_rechecked.json \
    ${STORAGE_PATH}/evaluation/*/results_gsm8k_*_hsp_rechecked.json \
    --output ${STORAGE_PATH}/summaries/hsp_grpo_final_summary.json
```

The summary reports:
- Per-action counts (ASK / VERIFY / ACCEPT rates)
- No-interaction vs any-interaction accuracy split
- Paired calibration (score delta vs independent baseline)
- Trust metrics (explicit accepts vs implicit adoptions)

---

## Project Structure

```
InfoBuy/
|
|-- README_HSP.md                          # >>> THIS FILE: HSP project entry point
|-- build_and_upload_hsp_dataset.py        # One-command data pipeline + HuggingFace upload
|
|-- SFT_stage/                             # SFT data construction and training
|   |-- fetch_hsp_source_dataset.py        # Fetch NuminaMath source pool with provenance
|   |-- build_hsp_source_splits.py         # Decontaminate + deterministic train/val split
|   |-- calibrate_hsp_difficulty.py        # Tag solve_rate + filter to ~70% difficulty band (GPU)
|   |-- build_hsp_sft.py                   # Generate 6 protocol sample types
|   |-- build_hsp_outcome_sft.py           # Select best rollout trajectories for replay SFT
|   |-- mix_hsp_sft.py                     # Mix protocol + replay data
|   |-- hsp_collator.py                    # Segment-aware collator with offset-based loss mask
|   |-- train_hsp.py                       # HSP SFT trainer with student-only loss
|   |-- preflight_hsp.py                   # Pre-training validation (dataset, tokenizer, config)
|   `-- tests/                             # Unit tests for all SFT components
|
|-- RL_stage/                              # Reinforcement learning
|   |-- examples/
|   |   |-- config_hsp.yaml                # Main HSP GRPO configuration
|   |   |-- config_hsp_shaped.yaml         # Shaped ablation configuration
|   |   |-- qwen3_hsp_grpo.sh             # HSP GRPO launcher (with preflight)
|   |   `-- reward_function/
|   |       `-- math_hsp_group.py          # Outcome-cost-trust reward function
|   |-- verl/
|   |   |-- workers/rollout/
|   |   |   `-- help_vllm_rollout_spmd.py  # HSP state machine rollout engine
|   |   `-- trainer/
|   |       `-- core_algos.py              # GRPO advantage estimation
|   |-- scripts/
|   |   `-- model_merger.py                # FSDP shard merger
|   `-- tests/                             # HSP rollout and reward tests
|
|-- eval/                                  # Evaluation
|   |-- generate_withhelp.py               # Collaborative generation (relay_call / hsp)
|   |-- evaluate_forhelp.bash              # Batch evaluation on 6 benchmarks
|   |-- collect_hsp_candidates.bash        # Counterfactual rollout collection
|   |-- summarize_hsp_results.py           # Action calibration summary
|   |-- results_recheck.py                 # Answer verification
|   |-- datasets_loader.py                 # Dataset handlers
|   `-- tests/                             # Evaluation tests
|
|-- utils/
|   `-- vllm_service.py                    # Teacher model Flask HTTP service
|
|-- ${INFOBUY_GENERATED_DATA}/                              # HSP data artifacts
|   |-- raw/                               # Source problems (train/val JSONL)
|   |-- protocol/                          # Protocol SFT data
|   |-- replay/                            # Outcome replay SFT data
|   `-- manifests/                         # Provenance and decontamination manifests
|
`-- docs/
    |-- hsp/                               # HSP design documentation
    |   |-- design.md                      # Full research design document
    |   |-- dataset_pipeline.md            # Dataset construction spec
    |   |-- sft_samples.md                 # The 6 SFT sample types, with examples
    |   |-- reward_design.md               # Outcome-cost-trust reward spec
    |   |-- rl_training_strategy.md        # GRPO training strategy
    |   `-- training_testing.md            # Operational checklist
    `-- legacy/                            # Baseline / upstream reference docs
        |-- uncertainty_trust_relay_implementation.md  # System-triggered baseline (comparison)
        `-- code_review_deepseek.md        # Historical code-review report
```

---

## Reward Function Design

The HSP reward function (`math_hsp_group.py`) optimizes a multi-objective signal:

```
overall = accuracy
        + useful_accept_weight   * useful_accepts          (reward)
        + resist_bad_review_weight * resisted_bad_reviews  (reward)
        + independent_correct_weight * no_interaction_correct (reward)
        - teacher_cost_weight    * (teacher_tokens / budget)  (cost penalty)
        - wrong_accept_weight    * wrong_accepts              (trust penalty)
        - implicit_adoption_weight * implicit_adoptions       (protocol penalty)
        - wrong_implicit_adoption_weight * wrong_implicit     (trust penalty)
        - unsupported_accept_weight * unsupported_accepts     (trust penalty)
        - invalid_accept_weight  * invalid_accepts            (protocol penalty)
        - invalid_protocol_weight * invalid_protocols         (protocol penalty)
        - denied_action_weight   * denied_actions             (budget penalty)
```

**Key reward components:**

| Component | Default Weight | What It Encourages/Penalizes |
|:----------|:--------------:|:-----------------------------|
| accuracy | 1.0 (base) | Correct final answer |
| teacher_cost | 0.15 | Penalizes excessive teacher token usage |
| wrong_accept | 0.50 | **Heavy penalty** for accepting incorrect teacher feedback |
| wrong_implicit_adoption | 0.50 | **Heavy penalty** for silently adopting wrong corrections |
| invalid_protocol | 0.20 | Penalizes forging environment markers |
| unsupported_accept | 0.10 | Penalizes accepting when teacher's correction is unverifiable |
| implicit_adoption | 0.05 | Mild penalty for using teacher's answer without `<ACCEPT>` |
| denied_action | 0.05 | Penalizes exceeding interaction budget |

The reward design philosophy: **correct trust is rewarded, misplaced trust is heavily penalized**, which teaches the student to be a critical consumer of external feedback.

---

## Configuration Reference

### `config_hsp.yaml` vs `config_hsp_shaped.yaml`

| Parameter | Main (`config_hsp`) | Shaped Ablation |
|:----------|:-------------------:|:---------------:|
| `useful_accept_weight` | 0.0 | 0.10 |
| `resist_bad_review_weight` | 0.0 | 0.10 |
| `teacher_error_weight` | 0.0 | 0.10 |
| `independent_correct_weight` | 0.0 | 0.05 |

The **main** config uses a minimal reward that relies on accuracy + cost + trust penalties. The **shaped ablation** adds explicit bonuses for desirable behaviors (useful accepts, resisting bad reviews, independent success).

### HSP Rollout Parameters

| Parameter | Description | Default |
|:----------|:------------|:-------:|
| `interaction_policy` | Must be `hsp` | `hsp` |
| `max_interactions` | Maximum ASK + VERIFY calls per rollout | 3 |
| `ask_budget_tokens` | Maximum teacher tokens per ASK | 64 |
| `verify_budget_tokens` | Maximum teacher tokens per VERIFY | 96 |
| `teacher_help_temperature` | Teacher sampling temperature for ASK | 0.7 |
| `teacher_review_temperature` | Teacher sampling temperature for VERIFY | 0.0 |

---

## Testing

Run all HSP-related tests (CPU only, no GPU required):

```bash
# SFT component tests
python -m unittest discover -s SFT_stage/tests -p 'test*.py'

# RL rollout state machine and reward tests
python -m unittest \
    RL_stage.tests.test_hsp_rollout_state \
    RL_stage.tests.test_math_hsp_reward

# Evaluation tests
python -m unittest \
    eval.tests.test_generate_hsp_segments \
    eval.tests.test_summarize_hsp_results \
    eval.tests.test_results_recheck

# Syntax check all HSP-related files
python -m py_compile SFT_stage/preflight_hsp.py
python -m py_compile SFT_stage/build_hsp_sft.py
python -m py_compile SFT_stage/build_hsp_outcome_sft.py
python -m py_compile SFT_stage/mix_hsp_sft.py
python -m py_compile SFT_stage/build_hsp_source_splits.py
python -m py_compile SFT_stage/fetch_hsp_source_dataset.py
python -m py_compile RL_stage/examples/reward_function/math_hsp_group.py
python -m py_compile eval/generate_withhelp.py
python -m py_compile eval/summarize_hsp_results.py

# Shell script syntax check
bash -n RL_stage/examples/qwen3_hsp_grpo.sh
bash -n eval/collect_hsp_candidates.bash
bash -n eval/evaluate_forhelp.bash
```

---

## Troubleshooting

### Preflight Validation Fails

```
ERROR: <ASK> is not a single tokenizer token
```
The SFT checkpoint was not trained with `train_hsp.py` or special tokens were not added. Retrain with `--add_context_tokens`.

```
ERROR: worker.rollout.interaction_policy must be hsp
```
You are using `config.yaml` instead of `config_hsp.yaml`. Switch to the HSP config.

```
ERROR: SFT max_seq_length=4096 is shorter than the RL visible sequence budget=12288
```
The SFT model was trained with a shorter context than RL requires. Retrain SFT with `--max_seq_length 12288` or reduce RL limits.

### Teacher Service Issues

```
ConnectionError: Connection refused on port 7778
```
The teacher service is not running. Start it with `python utils/vllm_service.py --model_path Qwen/Qwen3-8B --port 7778`.

### RL Training Issues

```
CUDA out of memory
```
Reduce `micro_batch_size_per_device_for_experience` (default 4, try 2) or `max_num_batched_tokens` (default 20000, try 16384).

```
ValueError: Please use reward_type=batch for the HSP reward function.
```
The config has `reward_type: function` instead of `reward_type: batch`. HSP reward requires batch mode to access cross-sample metadata.

### Evaluation Issues

```
ValueError: Dataset 'math' is wired to evaluation data and cannot be labeled as train.
```
You are trying to collect replay candidates from a benchmark dataset. Use `dataset=local_json` with `--name path/to/train.jsonl` instead.

```
OPENAI_API_KEY is required for answer recheck
```
Set `export SKIP_LLM_RECHECK=1` for deterministic-only evaluation, or provide an OpenAI API key.
