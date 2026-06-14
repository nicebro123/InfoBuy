# InfoBuy Experiment Specs

This directory contains reproducible HSP experiment specs. The public workflow
is:

```bash
bash run.sh smoke
bash run.sh teacher --gpu 1
bash run.sh rl-smoke --gpu 0 --teacher-gpus 1
bash run.sh train --spec configs/experiments/hsp_pilot.yaml --teacher-gpus 1 --gpu-pairs '0'
```

Launching HSP RL requires two distinct GPU roles: the already-running teacher
service and at least one training worker. Keep `--teacher-gpus` disjoint from
`--gpu-pairs`.

Specs are compact study files. `scripts/launch_hsp_experiments.py` expands each
experiment into immutable files under `$INFOBUY_STORE/experiments/<study>/`:

```text
launch_manifest.yaml
launch_tmux.sh
run_gpu0.sh
<run_name>/
  run_config.yaml
  train.log
  markers/
```

The launcher does not store data, weights, checkpoints, logs, or generated run
configs in the Git repo.

## Spec Fields

```yaml
study_name: hsp_pilot
base_config: RL_stage/examples/config_hsp.yaml
output_root: ${INFOBUY_STORE}/experiments
checkpoint_root: ${INFOBUY_CKPT}/rl
launcher: RL_stage/examples/qwen3_hsp_grpo.sh

defaults:
  gpu: "0"
  model_path: ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft
  teacher_port: 7778

experiments:
  - name: main
    run_name: qwen3_hsp_grpo_main
    config: RL_stage/examples/config_hsp.yaml
    overrides:
      trainer.experiment_name: qwen3_hsp_grpo_main
```

Dotted keys under `overrides` are applied to the materialized
`run_config.yaml`.

## Official Progression

| Spec | Purpose |
|---|---|
| `hsp_smoke.yaml` | 2-step GRPO runtime smoke after smoke data exists |
| `hsp_pilot.yaml` | main vs shaped pilot comparison |
| `hsp_ablation_cost.yaml` | teacher-cost ablations |
| `hsp_ablation_trust.yaml` | trust calibration ablations |
| `hsp_ablation_budget.yaml` | teacher token budget ablations |
| `hsp_ablation_interactions.yaml` | interaction-count ablations |
| `hsp_hparam_sweep.yaml` | KL and learning-rate sweep |
| `hsp_official.yaml` | combined official queue for main paper runs |
| `hsp_analysis.yaml` | RelayLLM-inspired reward component and fixed-budget analysis |

Generate scripts without launching:

```bash
python scripts/launch_hsp_experiments.py \
  --spec configs/experiments/hsp_pilot.yaml
```

Start tmux queues:

```bash
python scripts/launch_hsp_experiments.py \
  --spec configs/experiments/hsp_pilot.yaml \
  --launch-tmux
```

Use multiple GPU workers:

```bash
INFOBUY_GPU_PAIRS='0;2;3' \
python scripts/launch_hsp_experiments.py \
  --spec configs/experiments/hsp_official.yaml \
  --teacher-gpus 1 \
  --launch-tmux
```

Run the RelayLLM-inspired analysis queue on the minimum two-GPU layout:

```bash
TEACHER_GPU=0 WORKER_GPUS=1 SPEC=configs/experiments/hsp_analysis.yaml \
bash scripts/launch_hsp_full_stack.sh --skip-smoke
```

Run analysis evaluation suites after exporting a checkpoint to a
HuggingFace-loadable directory:

```bash
MODEL_PATH=/path/to/model GPU_QUEUE=1 bash scripts/run_hsp_analysis_eval.sh all
```
