#!/bin/bash
set -euo pipefail

export VLLM_DISABLE_COMPILE_CACHE=1

if [ "$#" -lt 4 ]; then
  echo "Usage: bash eval/evaluate_forhelp.bash <student_model> <teacher_name> <port> \"<gpu_ids>\" [interaction_policy] [samples_per_question] [collection_mode]" >&2
  echo "Optional env: EVAL_TASKS=\"math gsm8k\" MAX_EXAMPLES=10 OUTPUT_TAG=smoke SKIP_LLM_RECHECK=1" >&2
  exit 2
fi

model_name=$1
larger_model=$2
port=$3
gpu_queue=$4
GPU_QUEUE=($gpu_queue)
if [ ${#GPU_QUEUE[@]} -eq 0 ]; then
  echo "At least one GPU id is required in the fourth argument, for example: \"0\" or \"0 1\"." >&2
  exit 2
fi
interaction_policy=${5:-relay_call}
samples_per_question=${6:-1}
collection_mode=${7:-policy}
generate_extra_args=()
recheck_extra_args=()
if [ "${OVERWRITE_RESULTS:-0}" = "1" ]; then
  generate_extra_args+=(--overwrite_results)
fi
if [ "${OVERWRITE_RECHECK:-0}" = "1" ]; then
  recheck_extra_args+=(--overwrite_recheck)
fi
if [ -n "${OUTPUT_TAG:-}" ]; then
  generate_extra_args+=(--output_tag "${OUTPUT_TAG}")
  recheck_extra_args+=(--output_tag "${OUTPUT_TAG}")
fi
if [ -n "${MAX_EXAMPLES:-}" ]; then
  generate_extra_args+=(--max_examples "${MAX_EXAMPLES}")
fi
if [ -n "${MAX_INTERACTIONS:-}" ]; then
  generate_extra_args+=(--max_interactions "${MAX_INTERACTIONS}")
fi
if [ -n "${ASK_BUDGET_TOKENS:-}" ]; then
  generate_extra_args+=(--ask_budget_tokens "${ASK_BUDGET_TOKENS}")
fi
if [ -n "${VERIFY_BUDGET_TOKENS:-}" ]; then
  generate_extra_args+=(--verify_budget_tokens "${VERIFY_BUDGET_TOKENS}")
fi
if [ -n "${STUDENT_TEMPERATURE:-}" ]; then
  generate_extra_args+=(--student_temperature "${STUDENT_TEMPERATURE}")
fi
if [ -n "${TEACHER_HELP_TEMPERATURE:-}" ]; then
  generate_extra_args+=(--teacher_help_temperature "${TEACHER_HELP_TEMPERATURE}")
fi
if [ -n "${TEACHER_REVIEW_TEMPERATURE:-}" ]; then
  generate_extra_args+=(--teacher_review_temperature "${TEACHER_REVIEW_TEMPERATURE}")
fi
if [ "${SKIP_LLM_RECHECK:-0}" = "1" ]; then
  recheck_extra_args+=(--skip_llm_recheck)
elif [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is required for answer recheck; set SKIP_LLM_RECHECK=1 for deterministic-only evaluation." >&2
  exit 2
fi
MODEL_NAMES=(
  "$model_name"
)
if [ -n "${EVAL_TASKS:-}" ]; then
  read -r -a TASKS <<< "${EVAL_TASKS}"
else
  TASKS=(
    "math"
    "gsm8k"
    "minerva"
    "olympiad"
    "aime2024"
    "aime2025"
  )
fi

echo "Available GPUs: ${GPU_QUEUE[@]}"

running_gpus=()
running_pids=()
job_failed=0

start_job() {
  local gpu_id="$1"
  local model="$2"
  local task="$3"

  echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Start task [${task}] with model [${model}] on GPU [${gpu_id}] ..."

  # --- MODIFIED LINE ---
  if [ ${#generate_extra_args[@]} -gt 0 ]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python -m eval.generate_withhelp --small_model "${model}" --dataset "${task}" --larger_model "${larger_model}" --interaction_policy "${interaction_policy}" --collection_mode "${collection_mode}" --samples_per_question "${samples_per_question}" --large_model_url "http://127.0.0.1:${port}/generate" "${generate_extra_args[@]}" &
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python -m eval.generate_withhelp --small_model "${model}" --dataset "${task}" --larger_model "${larger_model}" --interaction_policy "${interaction_policy}" --collection_mode "${collection_mode}" --samples_per_question "${samples_per_question}" --large_model_url "http://127.0.0.1:${port}/generate" &
  fi

  running_gpus+=("${gpu_id}")
  running_pids+=("$!")
}

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "==> Processing model: ${MODEL_NAME}"
    TASK_INDEX=0
    NUM_TASKS=${#TASKS[@]}

    while :; do
        while [ ${#GPU_QUEUE[@]} -gt 0 ] && [ ${TASK_INDEX} -lt ${NUM_TASKS} ]; do
            gpu_id="${GPU_QUEUE[0]}"
            GPU_QUEUE=("${GPU_QUEUE[@]:1}")

            task="${TASKS[${TASK_INDEX}]}"
            TASK_INDEX=$((TASK_INDEX + 1))

            start_job "$gpu_id" "$MODEL_NAME" "$task"
        done

        if [ ${TASK_INDEX} -ge ${NUM_TASKS} ] && [ ${#running_pids[@]} -eq 0 ]; then
            break
        fi

        for running_index in "${!running_pids[@]}"; do
            gpu_id="${running_gpus[$running_index]}"
            pid="${running_pids[$running_index]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                if wait "$pid"; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] job finished with PID [${pid}]."
                else
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] job failed with PID [${pid}]." >&2
                    job_failed=1
                fi
                unset 'running_pids[running_index]'
                unset 'running_gpus[running_index]'
                GPU_QUEUE+=("$gpu_id")
            fi
        done

        sleep 1
    done
done

if [ "${job_failed}" -ne 0 ]; then
    echo "==> At least one evaluation job failed; result recheck was not started." >&2
    exit 1
fi

# --- MODIFIED LINES ---
if [ ${#recheck_extra_args[@]} -gt 0 ]; then
  python -m eval.results_recheck --model_name "$model_name" --larger_model "${larger_model}" --interaction_policy "${interaction_policy}" --collection_mode "${collection_mode}" "${recheck_extra_args[@]}"
else
  python -m eval.results_recheck --model_name "$model_name" --larger_model "${larger_model}" --interaction_policy "${interaction_policy}" --collection_mode "${collection_mode}"
fi


echo "==> All tasks have finished!"
