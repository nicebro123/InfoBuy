# InfoBuy Run Memory

This note records the working state established on 2026-06-14 for the
`slm` conda environment and the HSP training/evaluation workflow.

## Environment

- Repository: `/mnt/infini-data/test/quan_space/codespace/InfoBuy`
- Conda environment: `slm`
- External store default: `/mnt/infini-data/test/quan_space/codespace/InfoBuy_store`
- Student base model symlink: `$INFOBUY_PRETRAINED_MODELS/Qwen3-0.6B`
- Teacher model symlink: `$INFOBUY_TEACHER_MODELS/qwen3-8b-main`
- Main SFT checkpoint: `$INFOBUY_CKPT/sft/qwen3-0.6b-hsp-sft`
- RL smoke checkpoint output:
  `$INFOBUY_CKPT/rl/qwen3_hsp_grpo_smoke/global_step_2/actor/model_world_size_1_rank_0.pt`

Use:

```bash
conda run -n slm <command>
```

or activate `slm` manually before running repository commands.

## Runtime Defaults

This machine has CUDA runtime wheels but no full CUDA toolkit. vLLM V1 can
trigger FlashInfer JIT sampling and fail with:

```text
OSError: CUDA_HOME environment variable is not set.
```

The repository now defaults HSP vLLM entrypoints to:

```bash
VLLM_USE_V1=0
```

This is set in:

- `utils/vllm_service.py`
- `eval/generate_withhelp.py`
- `run.sh`
- `scripts/hsp_rollout_smoke.py`
- `scripts/launch_hsp_experiments.py`
- `RL_stage/examples/qwen3_hsp_grpo.sh`

Users can explicitly opt back into vLLM V1 by exporting `VLLM_USE_V1=1` in an
environment with a full CUDA toolkit.

## Verified Commands

Environment and tests:

```bash
conda run -n slm python -m pip check
conda run -n slm bash run.sh checks
```

Both passed. `run.sh checks` reported `113 tests OK`.

Smoke data and preflight:

```bash
conda run -n slm bash run.sh smoke
```

Small-model SFT smoke:

```bash
conda run -n slm bash run.sh sft-smoke --gpu 3 --foreground
```

This completed 2 training steps with Qwen3-0.6B.

Teacher service smoke:

```bash
conda run -n slm bash run.sh teacher --gpu 2 --port 7778 --foreground
```

Health request:

```bash
python - <<'PY'
import requests
payload = [{"prompt": "Solve: 2+3=", "max_tokens": 8, "temperature": 0.0}]
response = requests.post("http://127.0.0.1:7778/generate", json=payload, timeout=120)
print(response.status_code)
print(response.json())
PY
```

HSP rollout smoke:

```bash
conda run -n slm bash run.sh rollout-smoke --gpu 3 --port 7778 --output-tag stable_default
```

This passed both forced interaction modes:

- `force_ask_first`: `ask=1`, `teacher_tokens=48`
- `force_verify_after_draft`: `verify=1`, `teacher_tokens=64`

RL smoke:

```bash
conda run -n slm bash -lc 'source setup/env.sh >/dev/null && VLLM_USE_V1=0 python scripts/launch_hsp_experiments.py --spec configs/experiments/hsp_smoke.yaml --gpus 3 --teacher-gpus 2 --launch --overwrite --teacher-check-timeout 60'
```

This completed the 2-step GRPO smoke and wrote the `global_step_2` actor
checkpoint.

## Recommended GPU Layout

Use disjoint GPUs for teacher and training.

Example:

```bash
# Teacher on GPU 2
conda run -n slm bash run.sh teacher --gpu 2 --port 7778 --foreground

# Student/RL training on GPU 3
conda run -n slm bash run.sh rollout-smoke --gpu 3 --port 7778
```

For full experiment queues:

```bash
conda run -n slm bash run.sh train \
  --teacher-gpus 2 \
  --gpu-pairs '3;4;5;6;7' \
  --skip-smoke
```

Dry run first:

```bash
conda run -n slm bash run.sh train \
  --teacher-gpus 2 \
  --gpu-pairs '3;4;5;6;7' \
  --dry-run
```

## GitHub Sync

The repository was synchronized to:

```text
git@github.com-infobuy:nicebro123/InfoBuy.git
```

Current synchronized commit:

```text
fa4ad69 Stabilize HSP training and rollout runtime
```

SSH key alias used:

```text
Host github.com-infobuy
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_infobuy
  IdentitiesOnly yes
```

The public key was added to the GitHub repository deploy keys with write
access.

## Notes

- `decord==0.6.0` was removed from the local `slm` environment because it made
  `pip check` fail with a platform metadata error and repository code did not
  import it.
- `verl` editable dependency is installed outside the repository under:
  `$INFOBUY_TMP/pip-src/verl`
- Temporary teacher services should be stopped after tests to release GPU
  memory and ports.
