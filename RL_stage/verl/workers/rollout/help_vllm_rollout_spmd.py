import os
import re
import requests
import torch
import torch.distributed
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from contextlib import contextmanager
from collections import defaultdict
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams

# --- Framework-Specific Imports ---
# Placeholders with safe fallbacks
try:
    from ...protocol import DataProto
    from ...utils import torch_functional as VF
    from ...utils.torch_dtypes import PrecisionType
    from .base import BaseRollout
    from .config import RolloutConfig
except ImportError:
    print("Warning: Framework imports failed. Using dummy classes.")
    class BaseRollout: pass
    class DataProto: pass
    class RolloutConfig: pass
    class VF: pass
    class PrecisionType: pass


HSP_CONTEXT_MARKER_TOKENS = (
    "<TEACHER_HELP>",
    "</TEACHER_HELP>",
    "<TEACHER_REVIEW>",
    "</TEACHER_REVIEW>",
    "<ENVIRONMENT_NOTICE>",
    "</ENVIRONMENT_NOTICE>",
)
HSP_BOXED_ANSWER_PATTERN = re.compile(r"\\boxed\s*\{", flags=re.IGNORECASE)


def _student_context_marker_count(text: str) -> int:
    return sum(text.count(marker) for marker in HSP_CONTEXT_MARKER_TOKENS)


def _teacher_help_answer_leak(text: str) -> bool:
    lowered = text.lower()
    return bool(HSP_BOXED_ANSWER_PATTERN.search(text)) or "final answer" in lowered or "suggested answer" in lowered


def _repeat_interleave(value: Any, repeats: int) -> Any:
    """Utility to repeat values for list, tensor, or numpy array."""
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, list):
        return [item for item in value for _ in range(repeats)]
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    return value


class helpvLLMRollout(BaseRollout):
    """
    A vLLM rollout implementing a collaborative generation loop for text-only models.
    """
    
    # Status Constants
    STATUS_WAIT_SMALL = "waiting_for_small_model"
    STATUS_WAIT_LARGE = "waiting_for_large_model"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"

    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[Any] = None,
    ):
        print("Initializing Collaborative helpvLLMRollout (Text-Only, with n-sampling support)")
        super().__init__()
        
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not getattr(config, "disable_tqdm", False))

        # Collaborative Configuration
        self.max_turns = getattr(config, "max_turns", 3)
        self.interaction_policy = getattr(config, "interaction_policy", "relay_call")
        self.call_token_str = getattr(config, "call_token", "<call>")
        self.end_call_token_str = getattr(config, "end_call_token", "</call>")
        self.ask_token_str = getattr(config, "ask_token", "<ASK>")
        self.end_ask_token_str = getattr(config, "end_ask_token", "</ASK>")
        self.verify_token_str = getattr(config, "verify_token", "<VERIFY>")
        self.end_verify_token_str = getattr(config, "end_verify_token", "</VERIFY>")
        self.accept_token_str = getattr(config, "accept_token", "<ACCEPT>")
        self.max_interactions = getattr(config, "max_interactions", 3)
        self.ask_budget_tokens = getattr(config, "ask_budget_tokens", 64)
        self.verify_budget_tokens = getattr(config, "verify_budget_tokens", 96)
        self.teacher_help_temperature = getattr(config, "teacher_help_temperature", 0.7)
        self.teacher_review_temperature = getattr(config, "teacher_review_temperature", 0.0)
        
        # Pre-compile Regex for performance
        pattern_str = rf"{re.escape(self.call_token_str)}\s*(\d+)\s*{re.escape(self.end_call_token_str)}"
        self.call_tag_pattern = re.compile(pattern_str)
        # HSP action patterns: <ASK>N</ASK> and <VERIFY>N</VERIFY>
        self._compile_hsp_action_patterns()

        # Token Limits
        self.global_max_tokens = getattr(config, "global_max_tokens", 8192)
        self.base_small_model_max_tokens = getattr(config, "base_small_model_max_tokens", 8192)

        self._validate_config(config)
        self._init_vllm_engine(model_path, config)
        self._init_sampling_params(config)

    def _compile_hsp_action_patterns(self):
        """Compile <ASK>N</ASK> / <VERIFY>N</VERIFY> patterns from the token strings.

        Kept as a method so unit tests that build the rollout via __new__ (bypassing
        __init__) can reproduce the same patterns after setting the token strings.
        """
        end_ask = getattr(self, "end_ask_token_str", "</ASK>")
        end_verify = getattr(self, "end_verify_token_str", "</VERIFY>")
        self.end_ask_token_str = end_ask
        self.end_verify_token_str = end_verify
        self.ask_tag_pattern = re.compile(
            rf"{re.escape(self.ask_token_str)}\s*(\d+)\s*{re.escape(end_ask)}"
        )
        self.verify_tag_pattern = re.compile(
            rf"{re.escape(self.verify_token_str)}\s*(\d+)\s*{re.escape(end_verify)}"
        )

    def _validate_config(self, config: RolloutConfig):
        if self.interaction_policy not in {"relay_call", "hsp"}:
            raise ValueError(f"Unsupported interaction_policy: {self.interaction_policy}")
        if getattr(config, "tensor_parallel_size", 1) > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")
        
        needed_tokens = getattr(config, "prompt_length", 0) + getattr(config, "response_length", 0)
        if getattr(config, "max_num_batched_tokens", 0) < needed_tokens:
            raise ValueError("max_num_batched_tokens must be greater than prompt_length + response_length.")

    def _init_vllm_engine(self, model_path: str, config: RolloutConfig):
        """Initialize the vLLM engine."""
        prompt_len = getattr(config, "prompt_length", 1024)
        resp_len = getattr(config, "response_length", 1024)
        
        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=getattr(config, "trust_remote_code", False),
            load_format="dummy",
            dtype="bfloat16",
            seed=getattr(config, "seed", 42),
            max_model_len=getattr(config, "max_model_len", None) or (prompt_len + resp_len),
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=getattr(config, "tensor_parallel_size", 1),
            gpu_memory_utilization=getattr(config, "gpu_memory_utilization", 0.9),
            max_num_batched_tokens=getattr(config, "max_num_batched_tokens", 8192),
            disable_log_stats=getattr(config, "disable_log_stats", False),
            enforce_eager=getattr(config, "enforce_eager", False),
            disable_custom_all_reduce=True,
            enable_chunked_prefill=getattr(config, "enable_chunked_prefill", False),
            enable_sleep_mode=True,
        )
        self.inference_engine.sleep(level=1)

    def _init_sampling_params(self, config: RolloutConfig):
        """Setup default sampling parameters."""
        stop_tokens = (
            [self.end_ask_token_str, self.end_verify_token_str]
            if self.interaction_policy == "hsp"
            else [self.end_call_token_str]
        )
        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": True,
            'stop': stop_tokens,
            'include_stop_str_in_output': True,
            'temperature': 0.7,
        }
        # Merge config overrides
        default_params = SamplingParams()
        config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
        for key in config_dict.keys():
            if hasattr(default_params, key):
                sampling_kwargs[key] = getattr(config, key)
        if self.interaction_policy == "hsp":
            # Policy actions are registered special tokens and must survive detokenization.
            sampling_kwargs["skip_special_tokens"] = False
        
        self.small_model_sampling_params = SamplingParams(**sampling_kwargs)
        print(f"Collaborative Sampling Params: {self.small_model_sampling_params}")

    @contextmanager
    def update_sampling_params(self, **kwargs):
        """Context manager to dynamically update sampling parameters."""
        if self.interaction_policy == "hsp" and kwargs.get("skip_special_tokens", False):
            raise ValueError("HSP rollout requires skip_special_tokens=False so policy actions remain observable.")
        old_params = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.small_model_sampling_params, key):
                    old_params[key] = getattr(self.small_model_sampling_params, key)
                    setattr(self.small_model_sampling_params, key, value)
        yield
        for key, value in old_params.items():
            setattr(self.small_model_sampling_params, key, value)

    def _call_large_model_service_batch(self, payloads: List[Dict], request_ids: List[int]) -> List[Dict]:
        """Synchronously calls the large model API."""
        port = getattr(self.config, "port", 8000)
        url = f'http://127.0.0.1:{port}/generate'
        print(f"  [LM Batch Call IDs: {request_ids}] --> Delegating batch of {len(payloads)} requests...")

        def _create_error_response(msg: str) -> List[Dict]:
            return [{"text": f"[{msg}]", "finish_reason": "error"} for _ in payloads]

        try:
            response = requests.post(url, json=payloads, timeout=6000)
            if response.status_code == 200:
                results_data = response.json()
                if 'error' in results_data:
                    error_msg = results_data.get('error', 'Unknown server error')
                    print(f"  [Error] Server: {error_msg}")
                    return _create_error_response(f"Error: Server-side: {error_msg}")

                results = results_data.get('results', [])
                if len(results) != len(payloads):
                    print(f"  [Error] Mismatch count. Exp {len(payloads)}, got {len(results)}")
                    return _create_error_response(f"Mismatch in response count")
                
                print(f"  [LM Batch Call IDs: {request_ids}] <-- Success.")
                return results
            else:
                print(f"  [Error] Status {response.status_code}: {response.text}")
                return _create_error_response(f"Error: Status {response.status_code}")
        except Exception as e:
            print(f"  [Error] Connection failed: {e}")
            return _create_error_response(f"Error: Connection failed: {e}")

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Main orchestration loop."""
        
        with self.update_sampling_params(**prompts.meta_info):
            initial_input_ids = prompts.batch["input_ids"]
            initial_batch_size = initial_input_ids.size(0)
            n = self.small_model_sampling_params.n or 1 # Default to 1 if None

            # 1. Initialize Request Pool
            request_pool = self._init_request_pool(prompts, initial_batch_size, n)
            
            # 2. Main Generation Loop
            active_requests = len(request_pool)
            while active_requests > 0:
                # A. Small Model Step
                self._handle_small_model_step(request_pool)
                
                # B. Large Model Step
                self._handle_large_model_step(request_pool)

                # C. Check Active Status
                active_requests = sum(1 for req in request_pool 
                                      if req["status"] not in [self.STATUS_DONE, self.STATUS_FAILED])

            # 3. Finalize and Pack
            return self._pack_final_batch(prompts, request_pool, n, initial_batch_size)

    def _init_request_pool(self, prompts: DataProto, batch_size: int, n: int) -> List[Dict]:
        pool = []
        for i in range(batch_size):
            for j in range(n):
                pool.append({
                    "id": i * n + j,
                    "prompt_group_id": i,
                    "raw_prompt_ids": list(prompts.non_tensor_batch["raw_prompt_ids"][i]),
                    "current_solution_ids": [],
                    "status": self.STATUS_WAIT_SMALL,
                    "turn_count": 0,
                    "current_token_count": 0,
                    "delegate_info": None,
                    "final_response_ids": None,
                    "large_model_contributions": [],
                    "non_policy_contributions": [],
                    "allow_actions": True,
                    "pending_action": None,
                    "pending_requested_budget": None,
                    "pending_review_event": None,
                    "interaction_count": 0,
                    "ask_count": 0,
                    "verify_count": 0,
                    "accept_count": 0,
                    "invalid_accept_count": 0,
                    "invalid_protocol_count": 0,
                    "denied_action_count": 0,
                    "teacher_tokens_used": 0,
                    "student_output_for_grading": "",
                    "events": [],
                    "termination_reason": None,
                })
        return pool

    def _handle_small_model_step(self, request_pool: List[Dict]):
        """Filter requests waiting for small model and execute generation."""
        candidates = [req for req in request_pool if req["status"] == self.STATUS_WAIT_SMALL]
        if not candidates:
            return

        # Split into Turn 0 (potential n-sampling) and Turn > 0 (continuation)
        turn_zero_reqs = [req for req in candidates if req['turn_count'] == 0]
        turn_plus_reqs = [req for req in candidates if req['turn_count'] > 0]

        if turn_zero_reqs:
            self._process_turn_zero_batch(turn_zero_reqs)
        
        if turn_plus_reqs:
            self._process_turn_plus_batch(turn_plus_reqs)

    def _process_turn_zero_batch(self, reqs: List[Dict]):
        """Handle the first turn generation which supports beam/n sampling."""
        grouped_reqs = defaultdict(list)
        for req in reqs:
            grouped_reqs[req['prompt_group_id']].append(req)

        vllm_inputs = []
        input_to_reqs_map = []
        sampling_params_list = []

        for group_id in sorted(grouped_reqs.keys()):
            group = grouped_reqs[group_id]
            first_req = group[0]

            # Check Global Limits
            remaining = self.global_max_tokens - first_req['current_token_count']
            if remaining <= 0:
                self._force_done(group, "Reached GLOBAL_MAX_TOKENS")
                continue

            vllm_inputs.append({
                "prompt_token_ids": first_req['raw_prompt_ids'] + first_req['current_solution_ids']
            })
            input_to_reqs_map.append(group)
            
            # Setup Group Params
            params = self.small_model_sampling_params.clone()
            params.max_tokens = min(self.base_small_model_max_tokens, remaining)
            if self.interaction_policy == "hsp" and not first_req["allow_actions"]:
                params.stop = []
            # Ensure n matches request count to align outputs
            params.n = len(group) 
            sampling_params_list.append(params)

        if not vllm_inputs:
            return

        outputs = self.inference_engine.generate(vllm_inputs, sampling_params_list)
        
        for i, vllm_output in enumerate(outputs):
            associated_reqs = input_to_reqs_map[i]
            completions = vllm_output.outputs
            
            for req, completion in zip(associated_reqs, completions):
                self._update_request_state(req, completion)

    def _process_turn_plus_batch(self, reqs: List[Dict]):
        """Handle subsequent turns, typically single-path generation."""
        vllm_inputs = []
        input_map = []
        sampling_params_list = []

        for req in reqs:
            remaining = self.global_max_tokens - req['current_token_count']
            if remaining <= 0:
                self._force_done([req], "Reached GLOBAL_MAX_TOKENS")
                continue

            prompt_ids = req['raw_prompt_ids'] + req['current_solution_ids']
            # Safety truncate for context window
            context_limit = getattr(self.config, "max_model_len", None) or (
                self.config.prompt_length + self.config.response_length
            )
            if len(prompt_ids) > context_limit:
                prompt_ids = prompt_ids[-context_limit:]

            vllm_inputs.append({"prompt_token_ids": prompt_ids})
            input_map.append(req)

            params = self.small_model_sampling_params.clone()
            params.max_tokens = min(self.base_small_model_max_tokens, remaining)
            params.n = 1
            if self.interaction_policy == "hsp" and not req["allow_actions"]:
                params.stop = []
            sampling_params_list.append(params)

        if not vllm_inputs:
            return

        # Force n=1 for individual requests
        with self.update_sampling_params(n=1):
            outputs = self.inference_engine.generate(vllm_inputs, sampling_params_list)

        for i, vllm_output in enumerate(outputs):
            req = input_map[i]
            if vllm_output.outputs:
                self._update_request_state(req, vllm_output.outputs[0])

    def _update_request_state(self, req: Dict, completion: Any):
        """Core logic to process small model output and determine next state."""
        generated_text = completion.text
        generated_ids = completion.token_ids

        if self.interaction_policy == "hsp":
            self._update_hsp_request_state(req, generated_text, generated_ids)
            return

        req['current_token_count'] += len(generated_ids)
        req['turn_count'] += 1
        req['current_solution_ids'].extend(generated_ids)
        
        match = self.call_tag_pattern.search(generated_text)
        
        if match and req['turn_count'] < self.max_turns:
            requested_tokens = int(match.group(1))
            requested_tokens = max(requested_tokens, 0)
            requested_tokens = min(requested_tokens, self.base_small_model_max_tokens)
            # requested_tokens = 500
            remaining = self.global_max_tokens - req['current_token_count']

            if remaining <= 0:
                self._force_done([req], "Reached GLOBAL_MAX_TOKENS before LM call")
            elif requested_tokens == 0:
                print(f"  [Req {req['id']}] WAITING (0 tokens requested).")
                req['status'] = self.STATUS_WAIT_SMALL
            else:
                capped_tokens = min(requested_tokens, remaining)
                print(f"  [Req {req['id']}] Delegate {capped_tokens} tokens.")
                req['status'] = self.STATUS_WAIT_LARGE
                req['delegate_info'] = {
                    "max_tokens": capped_tokens,
                    "last_generated_token_count": len(generated_ids)
                }
        else:
            req['status'] = self.STATUS_DONE
            req['final_response_ids'] = req['current_solution_ids']
            if req['current_token_count'] >= self.global_max_tokens and completion.finish_reason == 'length':
                print(f"  [Req {req['id']}] Hit GLOBAL_MAX_TOKENS during generation.")

    def _update_hsp_request_state(self, req: Dict, generated_text: str, generated_ids: List[int]):
        req["current_token_count"] += len(generated_ids)
        req["turn_count"] += 1
        req["current_solution_ids"].extend(generated_ids)
        req["student_output_for_grading"] += generated_text
        self._record_hsp_review_decision(req, generated_text)
        req["invalid_protocol_count"] += _student_context_marker_count(generated_text)

        if not req["allow_actions"]:
            req["denied_action_count"] += (
                len(self.ask_tag_pattern.findall(generated_text))
                + len(self.verify_tag_pattern.findall(generated_text))
            )

        action = None
        requested_budget = None
        if req["allow_actions"]:
            ask_match = self.ask_tag_pattern.search(generated_text)
            verify_match = self.verify_tag_pattern.search(generated_text)
            if ask_match:
                action = "ask"
                requested_budget = int(ask_match.group(1))
            elif verify_match:
                action = "verify"
                requested_budget = int(verify_match.group(1))

        if action is None:
            req["status"] = self.STATUS_DONE
            req["final_response_ids"] = req["current_solution_ids"]
            return

        if req["interaction_count"] >= self.max_interactions:
            req["denied_action_count"] += 1
            notice = (
                "\n<ENVIRONMENT_NOTICE>No further external interactions are available. "
                "Complete the answer independently.</ENVIRONMENT_NOTICE>\n"
            )
            if not self._append_non_policy_text(req, notice):
                self._force_done([req], "Reached GLOBAL_MAX_TOKENS before interaction-limit notice")
                return
            req["allow_actions"] = False
            req["status"] = self.STATUS_WAIT_SMALL
            return

        req["interaction_count"] += 1
        req[f"{action}_count"] += 1
        req["pending_action"] = action
        req["pending_requested_budget"] = requested_budget
        req["events"].append({
            "action": action,
            "executed": True,
            "requested_budget_tokens": requested_budget,
            "student_before_feedback": req["student_output_for_grading"],
            "student_before_feedback_scope": "cumulative_student_visible",
            "teacher_text": None,
            "teacher_tokens_used": 0,
            "accepted": False if action == "verify" else None,
            "student_after_feedback": None,
            "observation_status": "pending",
            "error": None,
        })
        req["status"] = self.STATUS_WAIT_LARGE

    def _record_hsp_review_decision(self, req: Dict, generated_text: str):
        accept_count = generated_text.count(self.accept_token_str)
        pending_index = req.get("pending_review_event")
        if pending_index is None:
            req["invalid_accept_count"] += accept_count
            return
        event = req["events"][pending_index]
        event["student_after_feedback"] = generated_text
        event["accepted"] = accept_count > 0
        if accept_count:
            req["accept_count"] += 1
            req["invalid_accept_count"] += max(accept_count - 1, 0)
        req["pending_review_event"] = None

    def _append_non_policy_text(self, req: Dict, text: str) -> bool:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        remaining = self.global_max_tokens - req["current_token_count"]
        if len(ids) > remaining:
            return False
        start = len(req["current_solution_ids"])
        req["current_solution_ids"].extend(ids)
        req["current_token_count"] += len(ids)
        req["non_policy_contributions"].append({"start_token": start, "end_token": start + len(ids)})
        return True

    def _terminate_hsp_without_observation(self, req: Dict, reason: str):
        event = req["events"][-1]
        event["observation_status"] = "omitted_no_context_budget"
        event["terminal_without_observation"] = True
        req["pending_action"] = None
        self._force_done([req], reason)

    def _resume_after_hsp_teacher_failure(self, req: Dict):
        notice = (
            "\n<ENVIRONMENT_NOTICE>External response unavailable. "
            "Complete the answer independently.</ENVIRONMENT_NOTICE>\n"
        )
        if not self._append_non_policy_text(req, notice):
            self._terminate_hsp_without_observation(req, "Insufficient token budget for environment failure notice")
            return
        req["pending_action"] = None
        req["events"][-1]["observation_status"] = "delivered_environment_notice"
        req["status"] = self.STATUS_WAIT_SMALL

    def _build_hsp_teacher_prompt(self, req: Dict, action: str) -> str:
        transcript = self.tokenizer.decode(req["raw_prompt_ids"] + req["current_solution_ids"])
        if action == "ask":
            instruction = (
                "\n\nYou are an external mathematical reasoning consultant. The student requested help. "
                "Provide one short useful next step only. Do not produce <ASK>, <VERIFY>, or <ACCEPT>, "
                "and do not take over the final answer.\n"
            )
        else:
            instruction = (
                "\n\nYou are an external verifier. Review the student's current reasoning and tentative answer. "
                "Do not produce control markers. Respond only in this format:\n"
                "Verdict: correct | incorrect | uncertain\n"
                "Issue: <specific issue or None>\n"
                "Correction: <critical correction or None>\n"
                "Suggested answer: <boxed answer or None>\n"
            )
        return transcript + instruction

    def _handle_large_model_step(self, request_pool: List[Dict]):
        """Identify requests waiting for LM, prepare payloads, and process results."""
        batch = [req for req in request_pool if req["status"] == self.STATUS_WAIT_LARGE]
        if not batch:
            return

        if self.interaction_policy == "hsp":
            self._handle_hsp_large_model_step(batch)
            return

        # 1. Prepare Payloads
        payloads = []
        for req in batch:
            full_text = self.tokenizer.decode(req['raw_prompt_ids'] + req['current_solution_ids'])
            match = self.call_tag_pattern.search(full_text)
            
            if match:
                clean_prompt = full_text[:match.start()].strip()
            else:
                clean_prompt = self.call_tag_pattern.sub("", full_text).strip()
            
            payloads.append({
                "prompt": clean_prompt,
                "max_tokens": req['delegate_info']['max_tokens']
            })

        # 2. Call API
        request_ids = [req['id'] for req in batch]
        results = self._call_large_model_service_batch(payloads, request_ids)

        # 3. Process Results
        for req, result in zip(batch, results):
            self._apply_large_model_result(req, result)

    def _handle_hsp_large_model_step(self, batch: List[Dict]):
        payloads = []
        callable_batch = []
        for req in batch:
            action = req["pending_action"]
            wrapper = (
                "\n<TEACHER_HELP>\n\n</TEACHER_HELP>\n"
                if action == "ask"
                else "\n<TEACHER_REVIEW>\n\n</TEACHER_REVIEW>\n"
            )
            wrapper_tokens = len(self.tokenizer.encode(wrapper, add_special_tokens=False))
            remaining = self.global_max_tokens - req["current_token_count"] - wrapper_tokens
            if remaining <= 0:
                req["events"][-1]["error"] = "Insufficient token budget for teacher observation."
                self._terminate_hsp_without_observation(req, "Insufficient token budget for teacher observation")
                continue
            # Use student-requested budget N from <ASK>N</ASK> / <VERIFY>N</VERIFY>,
            # falling back to the config default if not available.
            student_requested = req.get("pending_requested_budget")
            if student_requested is not None and student_requested > 0:
                requested_budget = student_requested
            else:
                requested_budget = self.ask_budget_tokens if action == "ask" else self.verify_budget_tokens
            applied_budget = min(requested_budget, remaining)
            req["events"][-1]["applied_budget_tokens"] = applied_budget
            payloads.append({
                "prompt": self._build_hsp_teacher_prompt(req, action),
                "max_tokens": applied_budget,
                "temperature": self.teacher_help_temperature if action == "ask" else self.teacher_review_temperature,
                "top_p": 1.0,
            })
            callable_batch.append(req)

        if not callable_batch:
            return
        results = self._call_large_model_service_batch(payloads, [req["id"] for req in callable_batch])
        for req, result in zip(callable_batch, results):
            self._apply_hsp_teacher_result(req, result)

    def _apply_hsp_teacher_result(self, req: Dict, result: Dict):
        action = req["pending_action"]
        event_index = len(req["events"]) - 1
        event = req["events"][event_index]
        teacher_text = result.get("text", "")
        if result.get("finish_reason", "error") == "error" or teacher_text.startswith("[Error:"):
            event["error"] = teacher_text
            self._resume_after_hsp_teacher_failure(req)
            return

        teacher_ids = self.tokenizer.encode(teacher_text.strip(), add_special_tokens=False)
        teacher_cost = int(result.get("token_count", len(teacher_ids)))
        reserved_markers = (
            self.ask_token_str,
            self.verify_token_str,
            self.accept_token_str,
        ) + HSP_CONTEXT_MARKER_TOKENS
        reserved_marker = next((marker for marker in reserved_markers if marker in teacher_text), None)
        if reserved_marker is not None:
            req["teacher_tokens_used"] += teacher_cost
            event.update({
                "teacher_text": teacher_text,
                "teacher_tokens_used": teacher_cost,
                "error": f"Teacher response contained reserved protocol marker: {reserved_marker}.",
            })
            self._resume_after_hsp_teacher_failure(req)
            return

        if action == "ask" and _teacher_help_answer_leak(teacher_text):
            req["teacher_tokens_used"] += teacher_cost
            event.update({
                "teacher_text": teacher_text,
                "teacher_tokens_used": teacher_cost,
                "error": "Teacher help response disclosed a final-answer candidate.",
            })
            self._resume_after_hsp_teacher_failure(req)
            return

        open_tag, close_tag = (
            ("<TEACHER_HELP>", "</TEACHER_HELP>")
            if action == "ask"
            else ("<TEACHER_REVIEW>", "</TEACHER_REVIEW>")
        )
        wrapper_prefix = f"\n{open_tag}\n"
        wrapper_suffix = f"\n{close_tag}\n"
        prefix_ids = self.tokenizer.encode(wrapper_prefix, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(wrapper_suffix, add_special_tokens=False)
        remaining = self.global_max_tokens - req["current_token_count"] - len(prefix_ids) - len(suffix_ids)
        visible_teacher_ids = teacher_ids[: max(remaining, 0)]
        start = len(req["current_solution_ids"])
        observation_ids = prefix_ids + visible_teacher_ids + suffix_ids
        req["current_solution_ids"].extend(observation_ids)
        req["current_token_count"] += len(observation_ids)
        end = len(req["current_solution_ids"])
        req["teacher_tokens_used"] += teacher_cost
        contribution = {
            "start_token": start,
            "end_token": end,
            "action": action,
            "teacher_tokens_used": teacher_cost,
            "teacher_context_tokens": len(visible_teacher_ids),
            "feedback_truncated": len(visible_teacher_ids) < len(teacher_ids),
        }
        req["non_policy_contributions"].append(contribution)
        req["large_model_contributions"].append(contribution)
        event.update({
            "teacher_text": teacher_text,
            "teacher_tokens_used": teacher_cost,
            "teacher_context_text": self.tokenizer.decode(visible_teacher_ids),
            "teacher_context_tokens": len(visible_teacher_ids),
            "feedback_truncated": len(visible_teacher_ids) < len(teacher_ids),
            "observation_status": "delivered_teacher_observation",
        })
        if action == "verify":
            req["pending_review_event"] = event_index
        req["pending_action"] = None
        req["status"] = self.STATUS_WAIT_SMALL

    def _apply_large_model_result(self, req: Dict, result: Dict):
        """Apply the large model output to the request state, handling rollback."""
        text = result.get("text", "")
        reason = result.get("finish_reason", "error")

        if reason == "error" or text.startswith("[Error:"):
            req['status'] = self.STATUS_DONE
            req['final_response_ids'] = req['current_solution_ids']
            return

        # Rollback logic
        rollback_count = req['delegate_info'].get('last_generated_token_count', 0)
        if rollback_count > 0:
            req['current_solution_ids'] = req['current_solution_ids'][:-rollback_count]
            req['current_token_count'] -= rollback_count
        
        contribution_start = len(req['current_solution_ids'])
        lm_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        req['current_token_count'] += len(lm_ids)
        req['current_solution_ids'].extend(lm_ids)
        contribution_end = len(req['current_solution_ids'])
        
        req['large_model_contributions'].append({
            "start_token": contribution_start,
            "end_token": contribution_end
        })

        if reason == "stop":
            req['status'] = self.STATUS_DONE
            req['final_response_ids'] = req['current_solution_ids']
        elif reason == "length":
            if req['current_token_count'] >= self.global_max_tokens:
                self._force_done([req], "Reached GLOBAL_MAX_TOKENS after LM call")
            else:
                req['status'] = self.STATUS_WAIT_SMALL
        else:
            req['status'] = self.STATUS_WAIT_SMALL

    def _force_done(self, reqs: List[Dict], msg: str):
        """Helper to mark requests as done with an error message appended."""
        for req in reqs:
            req['status'] = self.STATUS_DONE
            req["termination_reason"] = msg
            if self.interaction_policy == "hsp":
                # Environment-side termination is metadata, not a student policy action.
                req['final_response_ids'] = req['current_solution_ids']
            else:
                error_ids = self.tokenizer.encode(f"\n[Info: {msg}]")
                req['final_response_ids'] = req['current_solution_ids'] + error_ids

    def _pack_final_batch(self, prompts: DataProto, request_pool: List[Dict], n: int, initial_batch_size: int) -> DataProto:
        """Construct the final DataProto output with padding and masking."""
        request_pool.sort(key=lambda r: r['id'])
        
        # Collect IDs
        resp_ids_list = [
            req['final_response_ids'] if req['final_response_ids'] is not None 
            else req['current_solution_ids'] 
            for req in request_pool
        ]
        
        # Pad Responses
        device = prompts.batch["input_ids"].device
        response_ids = VF.pad_2d_list_to_length(
            resp_ids_list, self.pad_token_id, max_length=self.config.response_length
        ).to(device)

        # Expand inputs if n > 1
        final_batch_size = initial_batch_size * n
        if n > 1:
            expanded_items = {
                k: _repeat_interleave(v, n) for k, v in prompts.batch.items()
            }
            batch = TensorDict(expanded_items, batch_size=final_batch_size)
        else:
            batch = prompts.batch.clone()

        # Build Masks and Position IDs
        sequence_ids = torch.cat([batch["input_ids"], response_ids], dim=-1)
        valid_response_mask = VF.get_response_mask(
            response_ids=response_ids, 
            eos_token_id=prompts.meta_info["eos_token_id"]
        )
        response_mask = valid_response_mask.clone()
        if self.interaction_policy == "hsp":
            for i, req in enumerate(request_pool):
                for contribution in req["non_policy_contributions"]:
                    start, end = contribution["start_token"], contribution["end_token"]
                    if start < response_mask.shape[1]:
                        response_mask[i, start : min(end, response_mask.shape[1])] = 0

        attention_mask = torch.cat((batch["attention_mask"], valid_response_mask), dim=-1)
        
        # Position IDs logic
        resp_len = response_ids.size(1)
        delta_pos = torch.arange(1, resp_len + 1, device=device).view(1, -1).expand(final_batch_size, -1)
        if batch["position_ids"].dim() == 3:
            delta_pos = delta_pos.view(final_batch_size, 1, -1).expand(final_batch_size, 3, -1)
        
        response_pos_ids = batch["position_ids"][..., -1:] + delta_pos
        position_ids = torch.cat([batch["position_ids"], response_pos_ids], dim=-1)

        # Consistency Check
        assert batch["input_ids"].shape[0] == final_batch_size, "Batch size mismatch"

        batch.update({
            "prompts": batch["input_ids"],
            "responses": response_ids,
            "input_ids": sequence_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "valid_response_mask": valid_response_mask,
            "position_ids": position_ids,
        })

        hsp_events = np.empty(len(request_pool), dtype=object)
        hsp_events[:] = [req["events"] for req in request_pool]
        student_outputs = np.empty(len(request_pool), dtype=object)
        student_outputs[:] = [
            req["student_output_for_grading"] if self.interaction_policy == "hsp" else None
            for req in request_pool
        ]
        full_transcripts = np.empty(len(request_pool), dtype=object)
        full_transcripts[:] = [
            self.tokenizer.decode(response_ids, skip_special_tokens=False) for response_ids in resp_ids_list
        ]
        non_tensor_batch = {
            "interaction_policy": np.array([self.interaction_policy for _ in request_pool], dtype=object),
            "hsp_events": hsp_events,
            "student_output_for_grading": student_outputs,
            "full_transcript": full_transcripts,
            "teacher_tokens_used": np.array([req["teacher_tokens_used"] for req in request_pool], dtype=object),
            "ask_count": np.array([req["ask_count"] for req in request_pool], dtype=object),
            "verify_count": np.array([req["verify_count"] for req in request_pool], dtype=object),
            "accept_count": np.array([req["accept_count"] for req in request_pool], dtype=object),
            "invalid_accept_count": np.array([req["invalid_accept_count"] for req in request_pool], dtype=object),
            "invalid_protocol_count": np.array([req["invalid_protocol_count"] for req in request_pool], dtype=object),
            "denied_action_count": np.array([req["denied_action_count"] for req in request_pool], dtype=object),
            "termination_reason": np.array([req["termination_reason"] for req in request_pool], dtype=object),
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
