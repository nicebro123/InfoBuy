#!/bin/bash

#BSUB -R "rusage[mem=100GB]"
#BSUB -gpu "num=4"
#BSUB -J qwen3_4b_math_grpo
#BSUB -o output.%J.log
#BSUB -e error.%J.log

set -x
# export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${INFOBUY_CKPT:-}" && -f "${REPO_ROOT}/setup/env.sh" ]]; then
    source "${REPO_ROOT}/setup/env.sh" >/dev/null
fi

CKPT_ROOT=${INFOBUY_CKPT:-${HSP_CKPT:-"${REPO_ROOT}/checkpoints"}}
MODEL_PATH=${MODEL_PATH:-${CKPT_ROOT}/sft/qwen3-0.6b-hsp-sft}
SAVE_PATH=${SAVE_PATH:-${CKPT_ROOT}/rl/group_new}
echo "MODEL_PATH=${MODEL_PATH}"
echo "SAVE_PATH=${SAVE_PATH}"
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=8192 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=group_new \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    worker.rollout.max_num_batched_tokens=20000 \
    worker.reward.reward_function=examples/reward_function/math_help_group.py:compute_score \
    worker.rollout.port=7778 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    trainer.find_last_checkpoint=false \
