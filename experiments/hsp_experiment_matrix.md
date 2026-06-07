# HSP Experiment Matrix

This matrix defines the required progression for InfoBuy experiments.

## Gate 0: Data And Reward Checks

Run before any GPU training:

```bash
python -m unittest RL_stage.tests.test_math_hsp_reward
python -m unittest SFT_stage.tests.test_preflight_hsp
bash experiments/run_hsp_smoke.sh
```

Required evidence:

- protocol smoke data passes preflight;
- `wrong_reject_count` is present in reward metrics;
- both main and shaped configs explicitly set every reward weight.

## Gate 1: Tiny Smoke

Purpose: prove the end-to-end runtime path on tiny data.

```bash
RUN_RL=1 bash experiments/run_hsp_smoke.sh
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
bash experiments/run_hsp_experiment.sh main
bash experiments/run_hsp_experiment.sh shaped
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
bash experiments/run_hsp_experiment.sh no_cost
bash experiments/run_hsp_experiment.sh cost_low
bash experiments/run_hsp_experiment.sh cost_high
bash experiments/run_hsp_experiment.sh trust_low
bash experiments/run_hsp_experiment.sh trust_high
bash experiments/run_hsp_experiment.sh budget_small
bash experiments/run_hsp_experiment.sh budget_large
bash experiments/run_hsp_experiment.sh interactions_1
bash experiments/run_hsp_experiment.sh interactions_4
```

Interpretation targets:

- cost ablations test whether the policy learns information purchase instead of always asking;
- trust ablations test whether ACCEPT behavior is calibrated;
- budget ablations test how much information should be purchased;
- interaction ablations test whether multiple purchases are necessary.

## Gate 4: Hyperparameter Sweeps

Run after the pilot and required ablations:

```bash
bash experiments/run_hsp_experiment.sh kl_low
bash experiments/run_hsp_experiment.sh kl_high
bash experiments/run_hsp_experiment.sh lr_low
bash experiments/run_hsp_experiment.sh lr_high
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
