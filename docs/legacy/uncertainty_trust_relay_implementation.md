# Uncertainty-Trust Relay Implementation Guide

> Status note (2026-05-25): This document describes a system-triggered uncertainty/trust baseline. The primary research direction for this repository is the student-controlled `<ASK>` / `<VERIFY>` / `<ACCEPT>` Help-Seeking Policy defined in `../hsp/design.md`. Keep this design as a comparison method rather than merging its trigger mechanism into the HSP main method.

This document is the implementation guide for changing RelayLLM from free-form textual calling into uncertainty-triggered, trust-aware collaborative decoding.

The goal is not to replace the whole project. Keep the current SFT stage and most of the EasyR1/verl training framework. Change only the rollout, evaluation decoding, metadata plumbing, and reward functions needed for the new paradigm.

## 1. Motivation

Current RelayLLM already lets the small model decide when to call the large model by generating:

```text
<call>N</call>
```

This solves the basic "when to call" problem, but the action is unconstrained:

- The model can call at arbitrary places.
- The model can request arbitrary token budgets.
- The rollout accepts the teacher response blindly.
- The difficulty-aware reward must prevent collapse into "always call" or "never call".

The new paradigm is:

```text
I am uncertain -> call the teacher with a controlled budget -> check whether the teacher response reduces my uncertainty
```

Two main ideas:

1. Self-uncertainty aware calling
   The system monitors the small model token distribution. A call is triggered by uncertainty signals such as low top-1 probability, small top-1/top-2 margin, or high approximate entropy. The small model no longer controls call length by emitting a number.

2. Trust-aware relay acceptance
   After the teacher generates a continuation, the small model evaluates whether the teacher continuation is plausible and whether it reduces uncertainty. This produces trust metrics used for logging and reward shaping.

## 2. Scope

Do not change the SFT stage for the first implementation.

Keep:

- `SFT_stage/train.py`
- `SFT_stage/add_command_upload.py`, except optional future cleanup of hard-coded tokens
- `utils/vllm_service.py`, except optional support for more sampling fields

Change:

- `eval/generate_withhelp.py`
- `RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py`
- `RL_stage/verl/workers/rollout/config.py`
- `RL_stage/verl/workers/reward/function.py`
- `RL_stage/examples/reward_function/math_help_group.py`

Optional cleanup:

- Fix `eval` vs `evaluation` package naming before running evaluation.
- Fix `RL_stage/script/model_merger.py` typo in `example.bash`.

## 3. New Decoding Policy

Replace textual call detection with uncertainty-triggered chunks.

Old behavior:

```text
small model generates until </call>
parse <call>N</call>
teacher generates N tokens
append teacher output
continue small model
```

New behavior:

```text
small model generates chunk_size tokens
collect per-token uncertainty from logprobs
if uncertainty crosses threshold:
    call teacher for fixed or rule-based budget
    score teacher response with trust metrics
    accept teacher response if policy allows
    continue small model
else:
    continue small model
```

The first version should use chunk-level control instead of token-by-token control. Token-by-token is more exact but much slower.

Recommended defaults:

```yaml
worker:
  rollout:
    uncertainty_call_enabled: true
    uncertainty_signal: "margin"       # "margin", "top1_prob", or "entropy"
    uncertainty_chunk_size: 32
    uncertainty_top_logprobs: 5
    uncertainty_margin_threshold: 0.15
    uncertainty_top1_threshold: 0.45
    uncertainty_entropy_threshold: 1.20
    uncertainty_min_uncertain_tokens: 3
    uncertainty_window_size: 8
    teacher_call_tokens: 64
    teacher_call_tokens_max: 256
    trust_enabled: true
    trust_accept_policy: "always"      # first implementation: always accept but record trust
    trust_probe_tokens: 16
    trust_min_gain: 0.0
```

For the first stable version, use:

- Trigger: `margin`
- Acceptance: `always`
- Reward: use trust as bonus/metric, not as hard rejection

This avoids early training instability from rejecting teacher text.

## 4. Data To Record

Rollout must record real collaborative process metadata. Do not infer call cost from text.

For each generated sample, record:

```python
{
    "num_calls": int,
    "teacher_tokens_requested": int,
    "teacher_tokens_used": int,
    "teacher_token_ratio": float,
    "accepted_teacher_tokens": int,
    "rejected_teacher_tokens": int,
    "mean_uncertainty_before_call": float,
    "mean_uncertainty_after_teacher": float,
    "mean_trust_gain": float,
    "mean_student_acceptance_logprob": float,
    "uncertainty_triggered": bool,
}
```

These fields must be returned from rollout through `DataProto.non_tensor_batch`. `ray_trainer.py` already unions rollout output into the full batch:

```python
new_batch = new_batch.union(gen_batch_output)
```

Therefore, if `gen_batch_output.non_tensor_batch` contains arrays with the same batch size as `responses`, reward functions can read them from `data.non_tensor_batch`.

## 5. File-by-file Implementation

### 5.1 `RL_stage/verl/workers/rollout/config.py`

Add fields to `RolloutConfig`.

Add:

```python
uncertainty_call_enabled: bool = False
uncertainty_signal: str = "margin"
uncertainty_chunk_size: int = 32
uncertainty_top_logprobs: int = 5
uncertainty_margin_threshold: float = 0.15
uncertainty_top1_threshold: float = 0.45
uncertainty_entropy_threshold: float = 1.20
uncertainty_min_uncertain_tokens: int = 3
uncertainty_window_size: int = 8
teacher_call_tokens: int = 64
teacher_call_tokens_max: int = 256
trust_enabled: bool = False
trust_accept_policy: str = "always"
trust_probe_tokens: int = 16
trust_min_gain: float = 0.0
```

Keep existing `port`.

Important: Some current fields are declared with `init=False`, including `port`. The CLI override `worker.rollout.port=7780` appears to work through OmegaConf conversion in the current project, so follow the local style unless it breaks. If the new fields cannot be overridden from YAML/CLI, remove `init=False` only from the new user-facing fields.

### 5.2 `RL_stage/verl/workers/rollout/help_vllm_rollout_spmd.py`

This is the main training rollout.

#### Keep

- `helpvLLMRollout`
- request pool structure
- large model service call through `/generate`
- final `DataProto` packing
- `large_model_contributions`

#### Stop relying on

- `self.call_tag_pattern.search(generated_text)` as the default call trigger
- `requested_tokens = int(match.group(1))` as the teacher budget

Do not delete the old textual call path immediately. Gate the new behavior behind:

```python
self.uncertainty_call_enabled = getattr(config, "uncertainty_call_enabled", False)
```

If disabled, old RelayLLM behavior should continue to run.

#### Add helper functions

Add these methods to `helpvLLMRollout`.

```python
def _init_uncertainty_config(self, config: RolloutConfig) -> None:
    ...
```

Initialize all uncertainty and trust config fields.

```python
def _build_small_sampling_params(self, max_tokens: int, n: int = 1) -> SamplingParams:
    ...
```

Return a fresh `SamplingParams` for each call. It should include:

- `max_tokens`
- `temperature`
- `top_p`
- `top_k`
- `n`
- `detokenize=True`
- `logprobs=self.uncertainty_top_logprobs` when uncertainty is enabled

For uncertainty mode, do not use `stop=["</call>"]`.

```python
def _extract_token_uncertainties(self, completion: Any) -> list[dict[str, float]]:
    ...
```

Extract uncertainty stats from vLLM completion logprobs.

Return one item per generated token:

```python
{
    "top1_prob": float,
    "top2_prob": float,
    "margin": float,
    "entropy": float,
}
```

Implementation notes:

- vLLM `CompletionOutput.logprobs` is usually a list, one entry per token.
- Each entry maps token id to logprob-like objects or floats depending on vLLM version.
- Write a small compatibility helper:

```python
def _logprob_value(x):
    return getattr(x, "logprob", x)
```

- Convert logprobs to probabilities with `np.exp`.
- Entropy from top-k is approximate:

```python
entropy = -sum(p_i * log(p_i + 1e-12) for p_i in probs)
```

This is a lower-bound entropy, but enough for triggering.

```python
def _should_call_from_uncertainty(self, stats: list[dict[str, float]]) -> tuple[bool, dict[str, float]]:
    ...
```

Use the last `uncertainty_window_size` tokens. Count uncertain tokens.

Rules:

- `margin`: uncertain if `margin < uncertainty_margin_threshold`
- `top1_prob`: uncertain if `top1_prob < uncertainty_top1_threshold`
- `entropy`: uncertain if `entropy > uncertainty_entropy_threshold`

Trigger if:

```python
num_uncertain >= uncertainty_min_uncertain_tokens
```

Return:

```python
(
    should_call,
    {
        "mean_margin": ...,
        "mean_top1_prob": ...,
        "mean_entropy": ...,
        "selected_uncertainty": ...,
        "num_uncertain": ...,
    }
)
```

```python
def _select_teacher_budget(self, uncertainty_summary: dict[str, float], remaining: int) -> int:
    ...
```

First version:

```python
return min(self.teacher_call_tokens, self.teacher_call_tokens_max, remaining)
```

Future version can scale budget by uncertainty severity.

```python
def _score_teacher_acceptance(self, req: dict, teacher_text: str, uncertainty_before: float) -> dict[str, float]:
    ...
```

First version can be lightweight:

- Tokenize current context + teacher text.
- Run a short small-model probe after teacher response.
- Compute uncertainty after teacher over `trust_probe_tokens`.
- `trust_gain = uncertainty_before - uncertainty_after`.

Do not implement full teacher-token logprob scoring in the first pass unless vLLM scoring is easy. The probe-based trust gain is enough for the first experiment.

```python
def _accept_teacher_response(self, trust_info: dict[str, float]) -> bool:
    ...
```

First version:

```python
if self.trust_accept_policy == "always":
    return True
if self.trust_accept_policy == "gain_threshold":
    return trust_info["trust_gain"] >= self.trust_min_gain
```

Recommended initial config is `"always"` to avoid destabilizing training.

```python
def _update_collab_metrics(self, req: dict, event: dict) -> None:
    ...
```

Maintain metrics on each request:

```python
req["collab_metrics"] = {
    "num_calls": 0,
    "teacher_tokens_requested": 0,
    "teacher_tokens_used": 0,
    "accepted_teacher_tokens": 0,
    "rejected_teacher_tokens": 0,
    "uncertainty_before_values": [],
    "uncertainty_after_values": [],
    "trust_gain_values": [],
    "student_acceptance_logprob_values": [],
}
```

Add this structure in `_init_request_pool`.

#### Modify `_init_sampling_params`

Current code uses:

```python
'stop': [self.end_call_token_str],
'include_stop_str_in_output': True,
```

In uncertainty mode, remove the call-token stop condition and set logprobs:

```python
if self.uncertainty_call_enabled:
    sampling_kwargs["max_tokens"] = self.uncertainty_chunk_size
    sampling_kwargs["logprobs"] = self.uncertainty_top_logprobs
    sampling_kwargs.pop("stop", None)
    sampling_kwargs.pop("include_stop_str_in_output", None)
```

For backward compatibility, keep old stop behavior when uncertainty is disabled.

#### Modify `_process_turn_zero_batch` and `_process_turn_plus_batch`

In uncertainty mode:

- Set `params.max_tokens` to `min(uncertainty_chunk_size, remaining)`.
- Set `params.logprobs`.
- Do not stop on `</call>`.
- After vLLM generation, call a new update method:

```python
self._update_request_state_uncertainty(req, completion)
```

Keep old `_update_request_state` for textual call mode.

Add:

```python
def _update_request_state_uncertainty(self, req: Dict, completion: Any):
    generated_ids = completion.token_ids
    req["current_solution_ids"].extend(generated_ids)
    req["current_token_count"] += len(generated_ids)
    req["turn_count"] += 1

    stats = self._extract_token_uncertainties(completion)
    should_call, summary = self._should_call_from_uncertainty(stats)

    if completion.finish_reason == "stop":
        req["status"] = self.STATUS_DONE
        req["final_response_ids"] = req["current_solution_ids"]
        return

    if req["current_token_count"] >= self.global_max_tokens:
        self._force_done([req], "Reached GLOBAL_MAX_TOKENS")
        return

    if should_call and req["turn_count"] < self.max_turns:
        remaining = self.global_max_tokens - req["current_token_count"]
        budget = self._select_teacher_budget(summary, remaining)
        req["status"] = self.STATUS_WAIT_LARGE
        req["delegate_info"] = {
            "max_tokens": budget,
            "uncertainty_before": summary["selected_uncertainty"],
            "uncertainty_summary": summary,
            "last_generated_token_count": 0,
        }
    else:
        req["status"] = self.STATUS_WAIT_SMALL
```

Important: In textual mode the old implementation rolls back the last small-model segment before applying teacher output. In uncertainty mode do not roll back by default. The uncertainty signal is observed after a chunk. The teacher should continue from the current state.

If you want teacher to replace the uncertain chunk later, add a separate `teacher_replace_uncertain_chunk` config. Do not add replacement in the first version.

#### Modify `_handle_large_model_step`

Current code removes `<call>N</call>` from the prompt. In uncertainty mode, there is no call tag.

Use:

```python
if self.uncertainty_call_enabled:
    clean_prompt = self.tokenizer.decode(req["raw_prompt_ids"] + req["current_solution_ids"])
else:
    old call-tag cleanup
```

Payload:

```python
{
    "prompt": clean_prompt,
    "max_tokens": req["delegate_info"]["max_tokens"],
    "temperature": getattr(self.config, "teacher_temperature", 0.7),
    "top_p": getattr(self.config, "teacher_top_p", 1.0),
}
```

Teacher generation can remain served by `utils/vllm_service.py`.

#### Modify `_apply_large_model_result`

In uncertainty mode:

- Do not rollback.
- Tokenize teacher text.
- Compute trust info if enabled.
- Accept or reject according to policy.
- Record metrics either way.

Pseudo-code:

```python
def _apply_large_model_result(self, req, result):
    if not self.uncertainty_call_enabled:
        return self._apply_large_model_result_textual(req, result)

    text = result.get("text", "")
    reason = result.get("finish_reason", "error")
    if reason == "error" or text.startswith("[Error:"):
        req["status"] = self.STATUS_WAIT_SMALL
        return

    lm_ids = self.tokenizer.encode(text, add_special_tokens=False)
    trust_info = self._score_teacher_acceptance(
        req, text, req["delegate_info"].get("uncertainty_before", 0.0)
    ) if self.trust_enabled else {}

    accepted = self._accept_teacher_response(trust_info)
    req["collab_metrics"]["num_calls"] += 1
    req["collab_metrics"]["teacher_tokens_requested"] += req["delegate_info"]["max_tokens"]
    req["collab_metrics"]["teacher_tokens_used"] += len(lm_ids)
    req["collab_metrics"]["uncertainty_before_values"].append(...)
    req["collab_metrics"]["uncertainty_after_values"].append(...)
    req["collab_metrics"]["trust_gain_values"].append(...)

    if accepted:
        start = len(req["current_solution_ids"])
        req["current_solution_ids"].extend(lm_ids)
        req["current_token_count"] += len(lm_ids)
        end = len(req["current_solution_ids"])
        req["large_model_contributions"].append({"start_token": start, "end_token": end})
        req["collab_metrics"]["accepted_teacher_tokens"] += len(lm_ids)
    else:
        req["collab_metrics"]["rejected_teacher_tokens"] += len(lm_ids)

    if reason == "stop" and accepted:
        req["status"] = self.STATUS_DONE
        req["final_response_ids"] = req["current_solution_ids"]
    elif req["current_token_count"] >= self.global_max_tokens:
        self._force_done([req], "Reached GLOBAL_MAX_TOKENS after teacher call")
    else:
        req["status"] = self.STATUS_WAIT_SMALL
```

To keep the diff clean, rename the current `_apply_large_model_result` to `_apply_large_model_result_textual` and introduce a new dispatcher `_apply_large_model_result`.

#### Modify `_pack_final_batch`

Currently returns:

```python
return DataProto(batch=batch, non_tensor_batch={}, meta_info=prompts.meta_info)
```

Change it to include collaboration metrics:

```python
non_tensor_batch = self._pack_collab_metrics(request_pool)
return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
```

Add:

```python
def _pack_collab_metrics(self, request_pool: list[dict]) -> dict[str, np.ndarray]:
    ...
```

Each returned value must be a numpy array with dtype `object` and length equal to final batch size.

Compute:

```python
teacher_token_ratio = teacher_tokens_used / max(response_token_count, 1)
mean_trust_gain = mean(trust_gain_values) or 0.0
mean_uncertainty_before_call = mean(uncertainty_before_values) or 0.0
mean_uncertainty_after_teacher = mean(uncertainty_after_values) or 0.0
```

Return keys:

```python
{
    "num_calls": np.array(..., dtype=object),
    "teacher_tokens_requested": np.array(..., dtype=object),
    "teacher_tokens_used": np.array(..., dtype=object),
    "teacher_token_ratio": np.array(..., dtype=object),
    "accepted_teacher_tokens": np.array(..., dtype=object),
    "rejected_teacher_tokens": np.array(..., dtype=object),
    "mean_uncertainty_before_call": np.array(..., dtype=object),
    "mean_uncertainty_after_teacher": np.array(..., dtype=object),
    "mean_trust_gain": np.array(..., dtype=object),
    "mean_student_acceptance_logprob": np.array(..., dtype=object),
    "uncertainty_triggered": np.array(..., dtype=object),
}
```

### 5.3 `RL_stage/verl/workers/reward/function.py`

Reward manager currently builds reward inputs with:

```python
{
    "prompt": prompt_str,
    "response": response_str,
    "response_length": cur_response_length,
    "ground_truth": ...
}
```

Add optional collaboration metadata. Do not assume every rollout has these fields.

Add helper:

```python
def _get_optional_non_tensor(data: DataProto, key: str, i: int, default):
    if key in data.non_tensor_batch:
        return data.non_tensor_batch[key][i]
    return default
```

Then include:

```python
"num_calls": _get_optional_non_tensor(data, "num_calls", i, 0),
"teacher_tokens_used": _get_optional_non_tensor(data, "teacher_tokens_used", i, 0),
"teacher_token_ratio": _get_optional_non_tensor(data, "teacher_token_ratio", i, 0.0),
"accepted_teacher_tokens": _get_optional_non_tensor(data, "accepted_teacher_tokens", i, 0),
"rejected_teacher_tokens": _get_optional_non_tensor(data, "rejected_teacher_tokens", i, 0),
"mean_uncertainty_before_call": _get_optional_non_tensor(data, "mean_uncertainty_before_call", i, 0.0),
"mean_uncertainty_after_teacher": _get_optional_non_tensor(data, "mean_uncertainty_after_teacher", i, 0.0),
"mean_trust_gain": _get_optional_non_tensor(data, "mean_trust_gain", i, 0.0),
"mean_student_acceptance_logprob": _get_optional_non_tensor(data, "mean_student_acceptance_logprob", i, 0.0),
"uncertainty_triggered": _get_optional_non_tensor(data, "uncertainty_triggered", i, False),
```

Do this in both `BatchFunctionRewardManager` and, if convenient, `SequentialFunctionRewardManager`. The current RelayLLM math reward uses batch mode, so batch mode is required.

### 5.4 `RL_stage/examples/reward_function/math_help_group.py`

Change reward from textual call parsing to actual teacher usage.

Old:

```python
if_call = '<call>' in response
call_num = sum(min(int(num), 4096) for num in re.findall(...))
call_ratio = min(call_num / reward_input["response_length"], 1.0)
```

New:

```python
teacher_tokens_used = int(reward_input.get("teacher_tokens_used", 0))
teacher_token_ratio = float(reward_input.get("teacher_token_ratio", 0.0))
used_teacher = teacher_tokens_used > 0
trust_gain = float(reward_input.get("mean_trust_gain", 0.0))
```

Keep the difficulty-aware group logic, but redefine the categories:

- `has_no_teacher_correct`: at least one sample correct with `used_teacher == False`
- `has_teacher_correct`: at least one sample correct with `used_teacher == True`

Recommended reward:

```python
cost_penalty = teacher_token_ratio
trust_bonus = max(0.0, min(trust_gain, 1.0)) * trust_weight
```

Use:

```python
if has_no_teacher_correct:
    if is_correct and not used_teacher:
        overall_score = 1.5
    elif is_correct and used_teacher:
        overall_score = 1.0 - cost_weight * cost_penalty + trust_bonus
    else:
        overall_score = 0.0

elif has_teacher_correct:
    if is_correct and used_teacher:
        overall_score = 1.0 - cost_weight * cost_penalty + trust_bonus
    elif not is_correct and not used_teacher:
        overall_score = -1.0
    elif not is_correct and used_teacher:
        overall_score = -0.2 * cost_penalty

else:
    if used_teacher and trust_gain > 0:
        overall_score = min(0.2, trust_bonus)
    else:
        overall_score = 0.0
```

Function signature:

```python
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    cost_weight: float = 1.0,
    trust_weight: float = 0.1,
) -> List[Dict[str, float]]:
```

Return metrics:

```python
{
    "overall": overall_score,
    "format": format_score,
    "mean@32": accuracy_score,
    "pass@32": pass_at_n_score,
    "teacher_token_ratio": teacher_token_ratio,
    "teacher_tokens_used": teacher_tokens_used,
    "num_calls": num_calls,
    "used_teacher": used_teacher,
    "trust_gain": trust_gain,
    "uncertainty_before": mean_uncertainty_before_call,
    "uncertainty_after": mean_uncertainty_after_teacher,
}
```

Do not remove `format_reward` or `accuracy_reward`.

### 5.5 `eval/generate_withhelp.py`

Implement the prototype here first. It is easier to debug than distributed training.

Current evaluation uses textual `<call>N</call>`. Add a CLI switch:

```python
parser.add_argument("--call_policy", type=str, default="textual", choices=["textual", "uncertainty"])
```

Add arguments:

```python
parser.add_argument("--uncertainty_signal", type=str, default="margin")
parser.add_argument("--chunk_size", type=int, default=32)
parser.add_argument("--top_logprobs", type=int, default=5)
parser.add_argument("--margin_threshold", type=float, default=0.15)
parser.add_argument("--top1_threshold", type=float, default=0.45)
parser.add_argument("--entropy_threshold", type=float, default=1.20)
parser.add_argument("--min_uncertain_tokens", type=int, default=3)
parser.add_argument("--window_size", type=int, default=8)
parser.add_argument("--teacher_call_tokens", type=int, default=64)
parser.add_argument("--trust_enabled", action="store_true")
parser.add_argument("--trust_accept_policy", type=str, default="always")
parser.add_argument("--trust_probe_tokens", type=int, default=16)
```

Add helper functions mirroring rollout:

- `extract_token_uncertainties(completion)`
- `should_call_from_uncertainty(stats, args)`
- `select_teacher_budget(summary, remaining, args)`
- `score_teacher_trust(llm, tokenizer, context, teacher_text, uncertainty_before, args)`
- `accept_teacher_response(trust_info, args)`

In uncertainty mode, sampling params:

```python
vllm.SamplingParams(
    temperature=0.7,
    max_tokens=min(args.chunk_size, remaining_tokens),
    logprobs=args.top_logprobs,
)
```

Output JSON should add:

```json
{
  "num_calls": 0,
  "teacher_tokens_used": 0,
  "teacher_token_ratio": 0.0,
  "mean_trust_gain": 0.0,
  "mean_uncertainty_before_call": 0.0,
  "mean_uncertainty_after_teacher": 0.0
}
```

Each per-item result file should have a sibling `results_*_summary.json` containing averaged:

- `avg_num_calls`
- `real_call_tokens`
- `real_call_tokens_ratio`
- `avg_trust_gain`

### 5.6 `eval/evaluate_forhelp.bash`

Add an optional mode argument:

```bash
call_policy=${5:-textual}
```

Pass:

```bash
--call_policy "${call_policy}"
```

Example:

```bash
bash eval/evaluate_forhelp.bash model_path 8B 7780 0 uncertainty
```

## 6. Trust Metric Details

Implement trust in two stages.

### Stage A: Probe-based trust gain

This is the first implementation.

Before teacher call:

```python
uncertainty_before = summary["selected_uncertainty"]
```

After teacher response is appended, ask small model to generate `trust_probe_tokens` more tokens with logprobs. Compute uncertainty over the probe.

```python
trust_gain = uncertainty_before - uncertainty_after
```

Interpretation:

- Positive: teacher response made the small model more confident.
- Zero or negative: teacher response did not help confidence.

Use this for metrics and reward bonus.

### Stage B: Student acceptance logprob

Add later if needed.

Score teacher tokens under the small model:

```text
mean log p_student(teacher_token | context)
```

This is more direct but requires an efficient scoring path. vLLM generation logprobs are easy for generated tokens, but scoring arbitrary teacher tokens may require a separate HF forward pass or vLLM prompt logprobs support. Do not block the first implementation on this.

## 7. Reward Redesign

Original difficulty-aware reward is still useful. Keep its group-level idea.

Change the variables:

```text
old: text contains <call>N</call>
new: rollout actually used teacher tokens
```

The new reward should optimize:

- Correctness
- Low teacher token cost
- Useful teacher help, measured by trust gain
- No teacher usage when student can solve the problem alone

Suggested formula is described in section 5.4.

Important warning:

Trust gain should be a small bonus, not the main objective. If it is too large, the model may learn to create artificial uncertainty before calling.

Recommended initial weights:

```yaml
worker:
  reward:
    reward_function_kwargs:
      cost_weight: 1.0
      trust_weight: 0.1
```

## 8. Backward Compatibility

The first PR should preserve old RelayLLM behavior.

Required gates:

- `uncertainty_call_enabled=False` means old textual `<call>N</call>` path.
- `call_policy=textual` means old evaluation behavior.
- Reward functions should default missing collaboration metadata to zero.

This lets us compare:

1. Original RelayLLM textual calling
2. Uncertainty-triggered calling
3. Uncertainty-triggered calling plus trust metrics
4. Uncertainty-triggered calling plus trust-aware reward

## 9. Implementation Order

Recommended order:

1. Evaluation prototype
   Modify `eval/generate_withhelp.py` only. Confirm uncertainty-triggered call works on a tiny dataset slice.

2. Evaluation logging
   Add teacher token ratio, uncertainty, and trust metrics to result JSON.

3. Rollout config
   Add new config fields to `RolloutConfig`.

4. Training rollout
   Add uncertainty mode to `help_vllm_rollout_spmd.py`.

5. DataProto metadata
   Return collaboration metrics in rollout `non_tensor_batch`.

6. Reward manager
   Pass collaboration metadata into `reward_inputs`.

7. Reward function
   Rewrite `math_help_group.py` to use real teacher token cost and trust gain.

8. Smoke tests
   Run unit tests that do not require GPU.

9. GPU smoke test
   Run one tiny rollout batch with `uncertainty_call_enabled=True`.

10. Full experiment
   Tune thresholds and weights.

## 10. Tests And Checks

Add or run these checks.

### Non-GPU checks

Run:

```bash
cd RL_stage
python -m pytest tests/test_dataproto.py tests/test_checkpoint.py
```

Add small pure tests if possible:

- `_should_call_from_uncertainty` returns true for low margins.
- `_should_call_from_uncertainty` returns false for confident stats.
- reward function handles missing metadata.
- reward function penalizes teacher usage when no-teacher correct exists.

### GPU smoke test

Use a very small local command:

```bash
CUDA_VISIBLE_DEVICES=1 python utils/vllm_service.py --model_path Qwen/Qwen3-8B --port 7780 &
cd RL_stage
CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.rollout_batch_size=2 \
  worker.rollout.n=2 \
  data.max_response_length=256 \
  worker.rollout.uncertainty_call_enabled=true \
  worker.rollout.uncertainty_chunk_size=16 \
  worker.rollout.teacher_call_tokens=32 \
  worker.rollout.trust_enabled=true \
  worker.rollout.port=7780 \
  trainer.max_steps=1 \
  trainer.val_before_train=false \
  trainer.save_freq=-1 \
  trainer.logger='["console"]'
```

Adjust model paths to local available models.

## 11. Expected Ablations

Use these experiment groups:

1. Base small model
2. Original RelayLLM textual call
3. Uncertainty-call only
4. Uncertainty-call plus trust metrics, always accept
5. Uncertainty-call plus trust-aware reward
6. Optional: trust threshold acceptance

Report:

- Accuracy
- Teacher token ratio
- Average calls per sample
- Average teacher tokens per call
- Mean trust gain
- Correlation between trust gain and correctness
- Accuracy by bins of pre-call uncertainty

## 12. Known Risks

1. vLLM logprobs API differences
   Handle both object-style logprobs and raw float logprobs.

2. Chunk-level trigger may call too late
   If accuracy drops, reduce `uncertainty_chunk_size` from 32 to 16 or 8.

3. Thresholds may be model-specific
   Tune thresholds per small model size.

4. Trust gain can be gamed
   Keep `trust_weight` small.

5. Hard rejection can hurt
   Start with `trust_accept_policy="always"`. Use trust for metrics and reward before using it to reject teacher responses.

6. Extra probe generation costs time
   Make `trust_enabled` optional. If training is too slow, disable trust during training and compute it in evaluation first.

## 13. Definition Of Done

The implementation is complete when:

- Textual RelayLLM mode still runs.
- Evaluation can run with `--call_policy uncertainty`.
- Training rollout can run with `worker.rollout.uncertainty_call_enabled=true`.
- Rollout returns teacher cost and trust metrics through `non_tensor_batch`.
- Reward function uses real teacher token usage instead of parsing `<call>N</call>`.
- Logs include teacher token ratio, number of calls, and trust gain.
- A tiny GPU smoke test completes one training step.
