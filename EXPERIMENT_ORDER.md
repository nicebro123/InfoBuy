# Experiment Order

Recommended order for the full InfoBuy HSP pipeline:

1. Full-data preflight:
   `python scripts/check_hsp_full_data.py --strict`
2. Protocol SFT and token probe:
   `bash run.sh sft --gpu 1`
   `bash run.sh token-probe --gpu 1`
3. Main paper RL queue:
   `configs/experiments/hsp_official.yaml`
4. RelayLLM-inspired analysis RL queue:
   `configs/experiments/hsp_analysis.yaml`
5. Full benchmark evaluation:
   `bash run.sh eval --gpu 1 --port 7778`
6. Analysis evaluation suites:
   `bash scripts/run_hsp_analysis_eval.sh all`

Core paper runs:

- `qwen3_hsp_grpo_main`: main HSP GRPO model.
- `qwen3_hsp_grpo_shaped`: shaped reward model.
- `qwen3_hsp_grpo_no_cost`: removes teacher-token cost.
- `qwen3_hsp_grpo_cost_005` / `qwen3_hsp_grpo_cost_030`: cost sensitivity.
- `qwen3_hsp_grpo_trust_025` / `qwen3_hsp_grpo_trust_080`: trust penalty sensitivity.
- `qwen3_hsp_grpo_budget_small` / `qwen3_hsp_grpo_budget_large`: teacher-token budget sensitivity.
- `qwen3_hsp_grpo_interactions_1` / `qwen3_hsp_grpo_interactions_4`: interaction-depth sensitivity.
- `qwen3_hsp_grpo_kl_003` / `qwen3_hsp_grpo_kl_030`: KL sensitivity.
- `qwen3_hsp_grpo_lr_3e-7` / `qwen3_hsp_grpo_lr_3e-6`: learning-rate sensitivity.

Analysis runs:

- `shaped_no_independent_bonus`: removes the independence incentive.
- `shaped_no_exploration_reward`: removes useful-accept and bad-review resistance rewards.
- `shaped_no_protocol_penalty`: removes invalid protocol and denied-action penalties.
- `shaped_no_accept_penalty`: removes wrong/unsupported accept penalties.
- `shaped_no_reject_penalty`: removes wrong-reject penalty.
- `shaped_no_teacher_error_penalty`: removes teacher-error penalty.
- `fixed_budget_32_32`, `fixed_budget_64_96`, `fixed_budget_128_192`: fixed ASK/VERIFY budgets.

All full pipeline commands are organized around the two-GPU layout:

- GPU 0: Qwen3-8B teacher service on port 7778.
- GPU 1: sequential student SFT/RL/eval worker queue.

The full SpecFlow-style pipeline queue is:

```bash
bash scripts/run_full_pipeline_2gpu.sh --launch
```
