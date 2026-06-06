# HSP Training and Testing Guide

This guide is the operational checklist for the Help-Seeking Policy (HSP) branch:

```text
student actions: <ASK>, <VERIFY>, <ACCEPT>
teacher observations: <TEACHER_HELP>, <TEACHER_REVIEW>
```

Run commands from the repository root unless a section explicitly says otherwise.

## 1. Environment

The lightweight data and protocol checks need standard Python packages used by this repo. Real SFT, vLLM evaluation, and GRPO require a training environment with:

```text
torch
datasets
transformers
omegaconf
ray
vllm
verl/EasyR1 runtime dependencies
```

For external answer recheck in batch evaluation, set:

```bash
export OPENAI_API_KEY=...
```

For deterministic smoke tests without external recheck:

```bash
export SKIP_LLM_RECHECK=1
```

## 2. Build Or Rebuild The Pilot Data

Fetch the NuminaMath-CoT synthetic_math source pool:

```bash
python -m SFT_stage.fetch_hsp_source_dataset \
  --max_records 1000 \
  --request_interval_seconds 2 \
  --output ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl \
  --manifest ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_seed_v0_1000.manifest.json
```

Split before protocol expansion to avoid same-problem leakage:

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

To reproduce the exact held-out snapshot already written beside the manifest, add:

```bash
--heldout_snapshot_input ${INFOBUY_GENERATED_DATA}/manifests/numinamath_cot_synthetic_math_split_pilot_v1.manifest.heldout_snapshot.json
```

Build protocol SFT data:

```bash
python -m SFT_stage.build_hsp_sft \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --variants_per_problem 1 \
  --seed 42

python -m SFT_stage.build_hsp_sft \
  --input ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_validation_pilot_v1.jsonl \
  --emit_all_types \
  --seed 42
```

Validate the structured protocol data:

```bash
python -m SFT_stage.preflight_hsp \
  --dataset ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --require_all_types \
  --require_context_tokens

python -m SFT_stage.preflight_hsp \
  --dataset ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_validation_pilot_v1.jsonl \
  --require_all_types \
  --require_context_tokens
```

## 3. SFT Warm-Up

Train the student with segment-level loss masking. Teacher observations stay visible context but do not enter labels.

```bash
python -m SFT_stage.train_hsp \
  --model_name Qwen/Qwen3-0.6B \
  --dataset ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --output_dir ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
  --max_seq_length 12288 \
  --bf16
```

After SFT, check tokenizer and length contracts:

```bash
python -m SFT_stage.preflight_hsp \
  --model_path ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
  --rl_config RL_stage/examples/config_hsp.yaml \
  --require_context_tokens
```

## 4. Teacher Service And Evaluation

Start the teacher service:

```bash
python utils/vllm_service.py \
  --model_path ${INFOBUY_TEACHER_MODELS}/qwen3-8b-main \
  --port 7778 \
  --tensor_parallel_size 1 \
  --trust_remote_code
```

Run one HSP evaluation:

```bash
python -m eval.generate_withhelp \
  --small_model ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
  --dataset math \
  --larger_model teacher_name \
  --large_model_url http://127.0.0.1:7778/generate \
  --interaction_policy hsp \
  --max_interactions 3 \
  --ask_budget_tokens 64 \
  --verify_budget_tokens 96 \
  --samples_per_question 8
```

Run batch evaluation:

```bash
bash eval/evaluate_forhelp.bash ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft teacher_name 7778 "0 1" hsp 8
```

Summarize HSP behavior:

```bash
python -m eval.summarize_hsp_results \
  ${STORAGE_PATH}/evaluation/*/results_math_*_hsp_rechecked.json \
  ${STORAGE_PATH}/evaluation/*/results_gsm8k_*_hsp_rechecked.json \
  --output ${STORAGE_PATH}/summaries/hsp_action_summary.json
```

## 5. Outcome Replay SFT

Collect counterfactual training candidates only from training data:

```bash
bash eval/collect_hsp_candidates.bash \
  ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft local_json teacher_name 7778 8 \
  --name ${INFOBUY_GENERATED_DATA}/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl \
  --output_tag pilot_r1 \
  --max_interactions 3 \
  --ask_budget_tokens 64 \
  --verify_budget_tokens 96
```

Build replay data from successful, low-cost, valid trajectories:

```bash
python -m SFT_stage.build_hsp_outcome_sft \
  --input \
    ${STORAGE_PATH}/evaluation/*/results_local_json_*_pilot_r1_hsp.json \
    ${STORAGE_PATH}/evaluation/*/results_local_json_*_pilot_r1_hsp_independent.json \
    ${STORAGE_PATH}/evaluation/*/results_local_json_*_pilot_r1_hsp_force_ask_first.json \
    ${STORAGE_PATH}/evaluation/*/results_local_json_*_pilot_r1_hsp_force_verify_after_draft.json \
  --output ${INFOBUY_GENERATED_DATA}/replay/hsp_outcome_replay_pilot_r1.jsonl \
  --teacher_token_budget 192 \
  --teacher_cost_weight 0.15 \
  --min_score 1.0

python -m SFT_stage.preflight_hsp \
  --dataset ${INFOBUY_GENERATED_DATA}/replay/hsp_outcome_replay_pilot_r1.jsonl

python -m SFT_stage.mix_hsp_sft \
  --protocol_data ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --replay_data ${INFOBUY_GENERATED_DATA}/replay/hsp_outcome_replay_pilot_r1.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/replay/hsp_sft_mixed_pilot_r1.jsonl \
  --max_replay_fraction 0.50
```

Use the mixed dataset for the next SFT round.

## 6. GRPO

The main HSP reward is in:

```text
RL_stage/examples/config_hsp.yaml
```

The shaped ablation is in:

```text
RL_stage/examples/config_hsp_shaped.yaml
```

Run main GRPO:

```bash
MODEL_PATH=${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
SAVE_PATH=${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_main \
bash RL_stage/examples/qwen3_hsp_grpo.sh
```

Run shaped ablation:

```bash
HSP_CONFIG=examples/config_hsp_shaped.yaml \
MODEL_PATH=${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
SAVE_PATH=${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_shaped \
bash RL_stage/examples/qwen3_hsp_grpo.sh
```

The launcher runs `SFT_stage/preflight_hsp.py` before training. It checks tokenizer tokens, SFT/RL length contract, GRPO config, response budget, and explicit reward weights.

## 7. Local Tests

Run deterministic HSP tests that do not need GPU:

```bash
python -m unittest discover -s SFT_stage/tests -p 'test*.py'

python -m unittest \
  RL_stage.tests.test_hsp_rollout_state \
  RL_stage.tests.test_math_hsp_reward \
  eval.tests.test_generate_hsp_segments \
  eval.tests.test_summarize_hsp_results \
  eval.tests.test_results_recheck

python -m unittest discover -s eval/tests -p 'test*.py'
```

Run syntax checks:

```bash
python -m py_compile \
  SFT_stage/preflight_hsp.py \
  SFT_stage/build_hsp_sft.py \
  SFT_stage/build_hsp_outcome_sft.py \
  SFT_stage/mix_hsp_sft.py \
  SFT_stage/build_hsp_source_splits.py \
  SFT_stage/fetch_hsp_source_dataset.py \
  RL_stage/examples/reward_function/math_hsp_group.py \
  RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py \
  eval/generate_withhelp.py \
  eval/summarize_hsp_results.py \
  eval/results_recheck.py

bash -n RL_stage/examples/qwen3_hsp_grpo.sh
bash -n eval/collect_hsp_candidates.bash
bash -n eval/evaluate_forhelp.bash
```

Full RL test discovery requires the real RL environment because several original EasyR1 tests import `torch` and `pytest`.

## 8. Current Known Boundary

The repository now contains the code path and reproducible pilot data for protocol SFT, candidate collection, replay construction, HSP reward, and GRPO launch. The remaining validation is runtime validation on a real training machine:

```text
1. Train a real HSP SFT checkpoint.
2. Start a real teacher vLLM service.
3. Run short HSP eval and candidate collection.
4. Run GRPO smoke test on GPU.
5. Compare main reward against shaped ablation and RelayLLM baseline.
```
