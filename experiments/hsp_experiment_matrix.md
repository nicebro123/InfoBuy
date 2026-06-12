# HSP Experiment Matrix

This matrix defines the required progression for InfoBuy experiments.

## Gate 0: Data And Reward Checks

Run before any GPU training:

```bash
python -m unittest RL_stage.tests.test_math_hsp_reward
python -m unittest SFT_stage.tests.test_preflight_hsp
bash run.sh smoke
```

Required evidence:

- protocol smoke data passes preflight;
- `wrong_reject_count` is present in reward metrics;
- both main and shaped configs explicitly set every reward weight.

## Gate 1: Tiny Smoke

Purpose: prove the end-to-end runtime path on tiny data.

```bash
bash run.sh teacher --gpu 1 --port 7778
bash run.sh rl-smoke --gpu 0
```

Expected settings:

- train questions: 16;
- validation questions: 8;
- rollout batch size: 8;
- GRPO group size: 2;
- max steps: 2;
- logger: console only.

Do not report research results from this gate.

## Gate 2: Pilot

Purpose: run a cheap but meaningful 800/200 pilot.

```bash
bash run.sh train \
  --spec configs/experiments/hsp_pilot.yaml \
  --gpu-pairs '0;1'
```

Required comparisons:

- answer accuracy;
- teacher token ratio;
- ASK rate;
- VERIFY rate;
- ACCEPT rate;
- wrong accept;
- wrong reject;
- implicit adoption.

## Gate 3: Required Ablations

Run each ablation with the same seed and data split as `main`.

```bash
bash run.sh train \
  --spec configs/experiments/hsp_ablation_cost.yaml \
  --gpu-pairs '0;1'

bash run.sh train \
  --spec configs/experiments/hsp_ablation_trust.yaml \
  --gpu-pairs '0;1'

bash run.sh train \
  --spec configs/experiments/hsp_ablation_budget.yaml \
  --gpu-pairs '0;1'

bash run.sh train \
  --spec configs/experiments/hsp_ablation_interactions.yaml \
  --gpu-pairs '0;1'
```

Interpretation targets:

- cost ablations test whether the policy learns information purchase instead of always asking;
- trust ablations test whether ACCEPT behavior is calibrated;
- budget ablations test how much information should be purchased;
- interaction ablations test whether multiple purchases are necessary.

## Gate 4: Hyperparameter Sweeps

Run after the pilot and required ablations:

```bash
bash run.sh train \
  --spec configs/experiments/hsp_hparam_sweep.yaml \
  --gpu-pairs '0;1'
```

To materialize the full official queue in one dry run:

```bash
bash run.sh train --dry-run --skip-smoke --gpu-pairs '0;1;2;3'
```

Generated run configs, queue scripts, logs, and markers live under:

```text
$INFOBUY_STORE/experiments/<study_name>/
```

Primary selection metric:

```text
accuracy - 0.15 * teacher_token_ratio
```

Secondary constraints:

- no collapse to always ASK;
- no collapse to never VERIFY;
- wrong accept and wrong reject both remain low;
- teacher token usage decreases versus unpriced or weak-cost settings.
