# InfoBuy

InfoBuy studies small-large model collaborative reasoning as an
information-buying problem. The student model learns:

```text
when to buy help
when to buy verification
how many teacher tokens to buy
whether to trust the teacher after verification
```

The current protocol is:

```text
<ASK>N</ASK>        buy bounded reasoning help
<VERIFY>N</VERIFY>  buy bounded verification
<ACCEPT>            explicitly accept verified feedback
```

This README is the reproducibility entry point. Detailed design notes are in
[`README_HSP.md`](README_HSP.md) and [`docs/hsp/`](docs/hsp/).

## Repository Layout

```text
InfoBuy/
├── configs/        experiment specs for smoke, pilot, ablations, sweeps
├── SFT_stage/      dataset builders, HSP collator, SFT trainer, preflight checks
├── RL_stage/       GRPO configs, HSP rollout state machine, reward function
├── eval/           HSP generation, benchmark evaluation, result recheck
├── experiments/    smoke tests, main runs, ablation launchers
├── scripts/        config-driven experiment and tmux launchers
├── setup/          external storage and download scripts
├── docs/hsp/       method, data, reward, training, storage docs
├── utils/          vLLM teacher service
└── run.sh          public reproduction entrypoint
```

Large files do not belong in this repository. Model weights, datasets,
checkpoints, logs, and evaluation outputs all live under an external store.

## 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/nicebro123/InfoBuy.git
cd InfoBuy

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Configure the external store. If `INFOBUY_STORE` is not set, `setup/env.sh`
defaults to a sibling directory such as `../InfoBuy_store`.

```bash
# Optional: choose a large disk explicitly.
export INFOBUY_STORE=$HOME/InfoBuy_store

source setup/env.sh
bash setup/make_dirs.sh
```

Important paths after sourcing `setup/env.sh`:

```text
$INFOBUY_STORE                 external root
$INFOBUY_PRETRAINED_MODELS     downloaded base model weights
$INFOBUY_TEACHER_MODELS        teacher aliases used by vLLM service
$INFOBUY_GENERATED_DATA        generated InfoBuy datasets
$INFOBUY_CKPT                  SFT, RL, and merged checkpoints
$STORAGE_PATH                  evaluation outputs
$HF_HOME                       Hugging Face cache on external store
```

Do not commit anything from `$INFOBUY_STORE`.

Run `source setup/env.sh` once in every new shell before using the training,
download, evaluation, or RL scripts.

## Quick Reproduction Path

After setup, the preferred public entrypoint is `run.sh`:

```bash
bash run.sh setup
bash run.sh download-models
bash run.sh download-data
bash run.sh build-data
bash run.sh smoke
bash run.sh sft --gpu 0
bash run.sh token-probe --gpu 0
bash run.sh teacher --gpu 1 --port 7778
bash run.sh rollout-smoke --gpu 0 --port 7778
bash run.sh rl-smoke --gpu 0 --teacher-gpus 1
bash run.sh train --teacher-gpus 1 --gpu-pairs '0'
```

Long-running commands start tmux sessions by default. Use `--foreground` when
you want to run in the current shell:

```bash
bash run.sh teacher --gpu 1 --foreground
```

HSP RL needs at least two distinct GPUs at launch time: one reserved for the
teacher service and at least one for RL training. Do not include the teacher GPU
inside `--gpu-pairs`. For example, if the teacher is on GPU `1`, use GPU `0`
for a two-GPU smoke or `0;2;3` for a larger queue.

The config-driven experiment system is:

```text
configs/experiments/*.yaml       compact study specs
scripts/launch_hsp_experiments.py materializes run_config.yaml + manifest
scripts/launch_hsp_tmux.sh        starts one study in tmux
scripts/launch_all_hsp_experiments.sh starts the official queue
```

Generated experiment artifacts are written outside Git:

```text
$INFOBUY_STORE/experiments/<study_name>/
├── launch_manifest.yaml
├── launch_tmux.sh
├── run_gpu0.sh
└── <run_name>/run_config.yaml
```

## 2. Download Model Weights

The default reproduction uses:

```text
student base: Qwen/Qwen3-0.6B
teacher:      Qwen/Qwen3-8B
optional:     Qwen/Qwen3-14B-Base
```

Download the student and teacher weights:

```bash
source setup/env.sh
bash setup/download_models.sh
```

This creates:

```text
$INFOBUY_PRETRAINED_MODELS/Qwen3-0.6B
$INFOBUY_PRETRAINED_MODELS/Qwen3-8B
$INFOBUY_TEACHER_MODELS/qwen3-8b-main -> ../pretrained/Qwen3-8B
```

To also download the optional stronger teacher:

```bash
source setup/env.sh
DOWNLOAD_OPTIONAL_TEACHERS=1 bash setup/download_models.sh
```

If Hugging Face access is required in your environment, run `hf auth login`
before downloading.

## 3. Download Reference Datasets

Download immutable source snapshots and cache evaluation datasets:

```bash
source setup/env.sh
bash setup/download_data.sh
```

Main source dataset:

```text
AI-MO/NuminaMath-CoT
```

Evaluation datasets cached or downloaded by the script:

```text
MATH-500 CSV
openai/gsm8k
zwhe99/amc23
zwhe99/simplerl-minerva-math
zwhe99/simplerl-OlympiadBench
HuggingFaceH4/aime_2024
yentinglin/aime_2025
```

DAPO is not part of the main InfoBuy training pipeline. Download it only for
optional RelayLLM baseline or ablation experiments:

```bash
source setup/env.sh
DOWNLOAD_DAPO_BASELINES=1 bash setup/download_data.sh
```

## 4. Build The HSP Dataset

The main training source is a decontaminated NuminaMath-CoT `synthetic_math`
pilot split. The protocol SFT data teaches the model the six behaviors needed
for information buying:

```text
normal
ask_help
verify_confirm
verify_accept_correction
verify_reject_bad_feedback
verify_uncertain
```

The default SFT training protocol emits all six behavior types for each
training problem. This is intentional: SFT is the protocol warm-up stage, so the
policy action tokens must be dense enough for the student to reliably learn
`<ASK>`, `<VERIFY>`, and `<ACCEPT>`. The RL stage then learns the calibrated
information-buying policy from the raw question/answer split.

Build the default pilot dataset:

```bash
source setup/env.sh

python build_and_upload_hsp_dataset.py \
  --build_only \
  --max_source_records 1000 \
  --train_size 800 \
  --val_size 200 \
  --seed 42
```

Expected outputs:

```text
$INFOBUY_GENERATED_DATA/raw/numinamath_cot_synthetic_math_seed_v0_1000.jsonl
$INFOBUY_GENERATED_DATA/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl
$INFOBUY_GENERATED_DATA/raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl
$INFOBUY_GENERATED_DATA/protocol/hsp_protocol_train_pilot_v1.jsonl
$INFOBUY_GENERATED_DATA/protocol/hsp_protocol_validation_pilot_v1.jsonl
$INFOBUY_GENERATED_DATA/manifests/*.json
```

Optional difficulty calibration needs a GPU and a reference model:

```bash
python build_and_upload_hsp_dataset.py \
  --build_only \
  --calibrate \
  --calibration_model ${INFOBUY_TEACHER_MODELS}/qwen3-8b-main
```

Validate the generated protocol data:

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

## 5. Run A Small Smoke Test First

Before full SFT or RL, run the lightweight smoke script. It builds a tiny local
slice and validates the protocol. If the pilot data is missing, it falls back to
the small test fixture.

```bash
source setup/env.sh
bash experiments/run_hsp_smoke.sh
```

Equivalent public entrypoint:

```bash
bash run.sh smoke
```

To also run a 2-step SFT smoke:

```bash
RUN_SFT=1 \
SFT_MODEL_NAME=${INFOBUY_PRETRAINED_MODELS}/Qwen3-0.6B \
bash experiments/run_hsp_smoke.sh
```

Or:

```bash
bash run.sh sft-smoke --gpu 0
```

To run the 2-step RL smoke, first start the teacher service on port `7778`
(section 7), then run:

```bash
source setup/env.sh
RUN_RL=1 bash experiments/run_hsp_smoke.sh
```

Preferred tmux/spec entrypoint:

```bash
bash run.sh rl-smoke --gpu 0 --teacher-gpus 1
```

## 6. Train The Student With SFT

SFT is the protocol warm-up stage. It teaches the student to use `<ASK>`,
`<VERIFY>`, and `<ACCEPT>` with teacher observations as context. Teacher tokens
are visible to the model but masked out of the training loss.

```bash
source setup/env.sh

python -m SFT_stage.train_hsp \
  --model_name ${INFOBUY_PRETRAINED_MODELS}/Qwen3-0.6B \
  --dataset ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --output_dir ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
  --max_seq_length 12288 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --bf16
```

Post-SFT compatibility check:

```bash
python -m SFT_stage.preflight_hsp \
  --model_path ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
  --rl_config RL_stage/examples/config_hsp.yaml \
  --require_context_tokens
```

SFT output:

```text
$INFOBUY_CKPT/sft/qwen3-0.6b-hsp-sft
```

Public tmux entrypoint:

```bash
bash run.sh sft --gpu 0
```

Before moving to RL, run the token probe. It checks that the SFT checkpoint has
registered the HSP action tokens and assigns `<ASK>`, `<VERIFY>`, and `<ACCEPT>`
reasonable next-token ranks in in-distribution protocol contexts:

```bash
bash run.sh token-probe --gpu 0
```

## 7. Start The Teacher Service

The teacher service answers HSP `<ASK>` and `<VERIFY>` calls during evaluation
and RL rollout.

```bash
source setup/env.sh

CUDA_VISIBLE_DEVICES=1 python utils/vllm_service.py \
  --model_path ${INFOBUY_TEACHER_MODELS}/qwen3-8b-main \
  --port 7778 \
  --tensor_parallel_size 1 \
  --trust_remote_code
```

Preferred tmux entrypoint:

```bash
bash run.sh teacher --gpu 1 --port 7778
```

Quick service check:

```bash
curl -s http://127.0.0.1:7778/generate \
  -H "Content-Type: application/json" \
  -d '[{"prompt": "What is 2+2?", "max_tokens": 32}]'
```

Keep this service running while using HSP evaluation or RL.

After the teacher is running, run a tiny HSP rollout smoke before GRPO. This
uses the smoke raw split and forced ASK/VERIFY collection modes to verify that
the student, teacher service, wrappers, event logging, and protocol validator
work together:

```bash
bash run.sh rollout-smoke --gpu 0 --port 7778
```

## 8. Evaluate The SFT Model

For deterministic local checks without external LLM recheck:

```bash
export SKIP_LLM_RECHECK=1
```

Single-dataset evaluation:

```bash
python -m eval.generate_withhelp \
  --small_model ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
  --dataset math \
  --larger_model Qwen3-8B \
  --large_model_url http://127.0.0.1:7778/generate \
  --interaction_policy hsp \
  --max_interactions 3 \
  --ask_budget_tokens 64 \
  --verify_budget_tokens 96 \
  --samples_per_question 8
```

Small batch benchmark smoke:

```bash
EVAL_TASKS="math gsm8k" \
MAX_EXAMPLES=10 \
OUTPUT_TAG=smoke \
SKIP_LLM_RECHECK=1 \
bash eval/evaluate_forhelp.bash \
  ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
  Qwen3-8B \
  7778 \
  "0" \
  hsp \
  2
```

Full batch evaluation:

```bash
bash eval/evaluate_forhelp.bash \
  ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft \
  Qwen3-8B \
  7778 \
  "0 1" \
  hsp \
  8
```

Evaluation outputs are written under:

```text
$STORAGE_PATH/evaluation/
$STORAGE_PATH/summaries/
```

Summarize HSP action behavior:

```bash
python -m eval.summarize_hsp_results \
  ${STORAGE_PATH}/evaluation/*/results_math_*_hsp_rechecked.json \
  ${STORAGE_PATH}/evaluation/*/results_gsm8k_*_hsp_rechecked.json \
  --output ${STORAGE_PATH}/summaries/hsp_action_summary.json
```

## 9. Optional Outcome Replay SFT

Outcome replay collects actual HSP rollouts, selects successful low-cost
trajectories, and mixes them back into the protocol SFT data.

Collect candidates from training data only:

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

Build and mix replay data:

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

python -m SFT_stage.mix_hsp_sft \
  --protocol_data ${INFOBUY_GENERATED_DATA}/protocol/hsp_protocol_train_pilot_v1.jsonl \
  --replay_data ${INFOBUY_GENERATED_DATA}/replay/hsp_outcome_replay_pilot_r1.jsonl \
  --output ${INFOBUY_GENERATED_DATA}/replay/hsp_sft_mixed_pilot_r1.jsonl \
  --max_replay_fraction 0.50
```

Then rerun SFT with the mixed dataset if you want a stronger warm-up.

## 10. Train With GRPO RL

RL learns the information-buying policy: when to ask, when to verify, how much
teacher budget to spend, and whether to accept feedback.

Make sure the teacher service from section 7 is still running.

Dry-run a spec first. This only writes immutable run configs and queue scripts;
it does not start training:

```bash
python scripts/launch_hsp_experiments.py \
  --spec configs/experiments/hsp_pilot.yaml
```

Inspect outputs under:

```text
$INFOBUY_STORE/experiments/hsp_pilot/
```

Smoke RL in tmux:

```bash
bash run.sh rl-smoke --gpu 0 --teacher-gpus 1
```

Pilot comparison in tmux:

```bash
bash run.sh train \
  --spec configs/experiments/hsp_pilot.yaml \
  --teacher-gpus 1 \
  --gpu-pairs '0'
```

Official queue in tmux:

```bash
bash run.sh train --teacher-gpus 1 --gpu-pairs '0;2;3'
```

This materializes and launches:

```text
configs/experiments/hsp_pilot.yaml
configs/experiments/hsp_ablation_cost.yaml
configs/experiments/hsp_ablation_trust.yaml
configs/experiments/hsp_ablation_budget.yaml
configs/experiments/hsp_ablation_interactions.yaml
configs/experiments/hsp_hparam_sweep.yaml
```

Single-study lower-level entrypoint:

```bash
bash scripts/launch_hsp_tmux.sh \
  --spec configs/experiments/hsp_ablation_cost.yaml \
  --teacher-gpus 1 \
  --gpu-pairs '0'
```

The experiment matrix is documented in
[`experiments/hsp_experiment_matrix.md`](experiments/hsp_experiment_matrix.md)
and [`configs/experiments/README.md`](configs/experiments/README.md).

Main RL checkpoints are saved under:

```text
$INFOBUY_CKPT/rl/qwen3_hsp_grpo_main
```

## 11. Merge And Evaluate RL Checkpoints

GRPO checkpoints are FSDP shards. Merge the actor checkpoint into Hugging Face
format before final evaluation:

```bash
source setup/env.sh
python RL_stage/scripts/model_merger.py \
  --local_dir ${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_main/actor
```

Merged model path:

```text
$INFOBUY_CKPT/rl/qwen3_hsp_grpo_main/actor/huggingface
```

Final evaluation:

```bash
source setup/env.sh
MERGED_MODEL=${INFOBUY_CKPT}/rl/qwen3_hsp_grpo_main/actor/huggingface

bash eval/evaluate_forhelp.bash \
  ${MERGED_MODEL} \
  Qwen3-8B \
  7778 \
  "0 1" \
  hsp \
  8
```

## 12. Recommended Reproduction Order

Use this order for a clean run:

```text
1. bash run.sh setup
2. bash run.sh download-models
3. bash run.sh download-data
4. bash run.sh build-data
5. bash run.sh smoke
6. bash run.sh sft --gpu 0
7. bash run.sh token-probe --gpu 0
8. bash run.sh teacher --gpu 1 --port 7778
9. bash run.sh rollout-smoke --gpu 0 --port 7778
10. bash run.sh eval --gpu 0
11. optional outcome replay SFT
12. bash run.sh rl-smoke --gpu 0 --teacher-gpus 1
13. bash run.sh train --teacher-gpus 1 --gpu-pairs '0'
14. merge RL actor checkpoint
15. run final evaluation and summarize results
```

## 13. Local Checks

Run these before committing or pushing:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s . -p 'test*.py'

python - <<'PY'
from pathlib import Path
path = Path("scripts/launch_hsp_experiments.py")
compile(path.read_text(encoding="utf-8"), str(path), "exec")
PY

bash -n \
  run.sh \
  setup/env.sh \
  setup/make_dirs.sh \
  setup/link_data.sh \
  setup/download_models.sh \
  setup/download_data.sh \
  experiments/run_hsp_smoke.sh \
  experiments/run_hsp_experiment.sh \
  scripts/launch_hsp_tmux.sh \
  scripts/launch_all_hsp_experiments.sh \
  eval/evaluate_forhelp.bash \
  RL_stage/examples/qwen3_hsp_grpo.sh

git diff --check
```

## 14. Troubleshooting

If `torch`, `vllm`, or `ray` is missing, install `requirements.txt` inside the
active environment.

If Hugging Face downloads fail, check:

```bash
hf auth login
echo $HF_HOME
echo $HF_HUB_CACHE
```

If evaluation says `OPENAI_API_KEY` is required, either set the key or run
deterministic-only checks:

```bash
export SKIP_LLM_RECHECK=1
```

If a script looks for `data/...` inside the repository, prefer updating the
script to use `$INFOBUY_GENERATED_DATA`. The compatibility bridge is:

```bash
bash setup/link_data.sh
```

This creates a local `data -> $INFOBUY_STORE/datasets` symlink, but it is not
required for the main pipeline.
