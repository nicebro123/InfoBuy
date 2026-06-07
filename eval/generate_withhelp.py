from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any

import requests
import vllm
from transformers import AutoTokenizer

try:
    from . import datasets_loader
except ImportError:
    import datasets_loader

STORAGE_PATH = os.getenv("STORAGE_PATH", ".")
# Remove global MAX_TURNS, manage or pass it through args in the function
GLOBAL_MAX_TOKENS = 8192
BENCHMARK_DATASETS = {
    "math", "gsm8k", "amc", "minerva", "olympiad",
    "aime2024", "aime2025", "mmlu_pro", "bbeh", "super_gpqa", "gpqa",
}
HSP_POLICY_ACTION_TOKENS = ("<ASK>", "</ASK>", "<VERIFY>", "</VERIFY>", "<ACCEPT>")
HSP_CONTEXT_MARKER_TOKENS = (
    "<TEACHER_HELP>",
    "</TEACHER_HELP>",
    "<TEACHER_REVIEW>",
    "</TEACHER_REVIEW>",
    "<ENVIRONMENT_NOTICE>",
    "</ENVIRONMENT_NOTICE>",
)
HSP_RESERVED_MARKERS = HSP_POLICY_ACTION_TOKENS + HSP_CONTEXT_MARKER_TOKENS
HSP_BOXED_ANSWER_PATTERN = re.compile(r"\\boxed\s*\{", flags=re.IGNORECASE)


def _user_prompt_content(question: str) -> str:
    return question.rstrip() + "\nPlease reason step by step, and put your final answer within \\boxed{}."


def _validate_data_role(dataset: str, data_role: str) -> None:
    if data_role == "train" and dataset in BENCHMARK_DATASETS:
        raise ValueError(
            f"Dataset '{dataset}' is wired to evaluation data and cannot be labeled as train. "
            "Use --dataset local_json --name /path/to/train.jsonl or a private training dataset."
        )


def _safe_output_component(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return rendered[:64] or "run"


def _result_dataset_key(dataset: str, dataset_name: str | None, output_tag: str | None) -> str:
    parts = [dataset]
    if dataset_name:
        identity = (
            str(Path(dataset_name).expanduser().resolve())
            if dataset == "local_json"
            else dataset_name
        )
        readable = Path(dataset_name).stem if dataset == "local_json" else dataset_name.rsplit("/", 1)[-1]
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
        parts.extend([_safe_output_component(readable), digest])
    if output_tag:
        parts.append(_safe_output_component(output_tag))
    return "_".join(parts)


def _result_output_path(args) -> str:
    if args.interaction_policy == "relay_call":
        policy_suffix = ""
    elif args.collection_mode == "policy":
        policy_suffix = "_hsp"
    else:
        policy_suffix = f"_hsp_{args.collection_mode}"
    output_dir = f"{STORAGE_PATH}/evaluation/{args.small_model.replace('/', '_')}_{args.larger_model}"
    dataset_key = _result_dataset_key(args.dataset, args.name, args.output_tag)
    return f"{output_dir}/results_{dataset_key}{policy_suffix}.json"


def _summary_output_path(result_output_path: str) -> str:
    path = Path(result_output_path)
    return str(path.with_name(f"{path.stem}_summary.json"))


def _reserved_hsp_marker(text: str) -> str | None:
    return next((marker for marker in HSP_RESERVED_MARKERS if marker in text), None)


def _student_context_marker_count(text: str) -> int:
    return sum(text.count(marker) for marker in HSP_CONTEXT_MARKER_TOKENS)


def _teacher_help_answer_leak(text: str) -> bool:
    lowered = text.lower()
    return bool(HSP_BOXED_ANSWER_PATTERN.search(text)) or "final answer" in lowered or "suggested answer" in lowered


def call_large_model_service_batch(payloads: List[Dict], request_ids: List[int], url: str) -> List[Dict]:
    print(f"  [LM Batch Call IDs: {request_ids}] --> Delegating batch of {len(payloads)} requests...")

    def _create_error_response(msg: str) -> List[Dict]:
        return [{"text": f"[{msg}]", "finish_reason": "error"}] * len(payloads)

    try:
        response = requests.post(url, json=payloads, timeout=6000)
        
        if response.status_code == 200:
            results_data = response.json()

            if 'error' in results_data:
                error_msg = results_data.get('error', 'Unknown server error')
                print(f"  [LM Batch Call IDs: {request_ids}] <-- Error: Server returned an error: {error_msg}")
                return _create_error_response(f"Error: Server-side: {error_msg}")

            generation_results = results_data.get('results', [])
            print(f"  [LM Batch Call IDs: {request_ids}] <-- Large model responded successfully.")

            if len(generation_results) != len(payloads):
                return _create_error_response(f"Error: Mismatch count. Exp {len(payloads)}, got {len(generation_results)}")
            
            return generation_results
        else:
            return _create_error_response(f"Error: Service status {response.status_code}")
            
    except Exception as e:
        print(f"  [LM Batch Call IDs: {request_ids}] <-- Connection Error: {e}")
        return _create_error_response(f"Error: Connection failed: {e}")


def run_relay_generation(llm: vllm.LLM, tokenizer: AutoTokenizer, prompts: List[str], large_model_url: str) -> List[Dict]:
    MAX_TURNS = 5
    BASE_SMALL_MODEL_MAX_TOKENS = GLOBAL_MAX_TOKENS

    call_tag_pattern = re.compile(r"<call>\s*\d+\s*</call>")

    request_pool: List[Dict] = [{
        "id": i, 
        "base_prompt": prompt, 
        "current_solution": "",
        "status": "waiting_for_small_model", 
        "turn_count": 0,
        "current_token_count": 0,
        "delegate_info": None, 
        "final_answer": None,
        "large_model_contributions": []
    } for i, prompt in enumerate(prompts)]

    active_requests = len(request_pool)
    
    while active_requests > 0:
        print(f"\n{'='*20} New Loop Iteration | Active Requests: {active_requests} {'='*20}")

        # --- Small Model Step ---
        small_model_batch = [req for req in request_pool if req["status"] == "waiting_for_small_model"]
        if small_model_batch:
            print(f"Found {len(small_model_batch)} requests for small model.")
            
            current_prompts = []
            small_model_params_list = []
            batch_to_run = [] 

            for req in small_model_batch:
                remaining_tokens = GLOBAL_MAX_TOKENS - req['current_token_count']

                if remaining_tokens <= 0:
                    req['status'] = 'done'
                    req['final_answer'] = req['current_solution'] + f"\n[Error: Reached GLOBAL_MAX_TOKENS ({GLOBAL_MAX_TOKENS})]"
                    continue

                current_prompts.append(req['base_prompt'] + req['current_solution'])
                
                turn_max_tokens = min(BASE_SMALL_MODEL_MAX_TOKENS, remaining_tokens)
                
                small_model_params_list.append(
                    vllm.SamplingParams(
                        temperature=0.7, 
                        max_tokens=turn_max_tokens,
                        stop=["</call>"], 
                        include_stop_str_in_output=True
                    )
                )
                batch_to_run.append(req)

            if batch_to_run:
                outputs = llm.generate(current_prompts, small_model_params_list)
                
                for req, output in zip(batch_to_run, outputs):
                    req['turn_count'] += 1
                    small_model_output = output.outputs[0].text
                    
                    generated_token_count = len(tokenizer.encode(small_model_output, add_special_tokens=False))
                    req['current_token_count'] += generated_token_count
                    
                    finish_reason = output.outputs[0].finish_reason

                    has_call_tag = "<call>" in small_model_output

                    if (finish_reason == "stop" and not has_call_tag) or req['turn_count'] >= MAX_TURNS:
                        # Normal termination or maximum number of turns reached
                        req['status'] = 'done'
                        req['final_answer'] = req['current_solution'] + small_model_output
                        req['current_solution'] += small_model_output # 保持一致性
                        print(f"  [Req {req['id']}] --> FINISHED (Reason: {finish_reason}, Turn: {req['turn_count']}).")
                        continue

                    if has_call_tag:
                        req['current_solution'] += small_model_output
                        # Extract token count
                        match = re.search(r"<call>\s*(\d+)", small_model_output)
                        if match:
                            requested_lm_tokens = int(match.group(1))
                            # requested_lm_tokens = 20
                            remaining_tokens_for_lm = GLOBAL_MAX_TOKENS - req['current_token_count']
                            
                            if remaining_tokens_for_lm <= 0:
                                req['status'] = 'done'
                                req['final_answer'] = req['current_solution'] + f"\n[Error: Limit Reached]"
                            elif requested_lm_tokens == 0:
                                req['status'] = 'waiting_for_small_model' # 0 token 视为不调用
                            else:
                                capped_lm_tokens = min(requested_lm_tokens, remaining_tokens_for_lm)
                                req['status'] = 'waiting_for_large_model'
                                req['delegate_info'] = {"max_tokens": capped_lm_tokens}
                                print(f"  [Req {req['id']}] --> DELEGATING (Tokens: {capped_lm_tokens}).")
                        else:
                            req['status'] = 'failed'
                            req['final_answer'] = req['current_solution'] + "\n[Error: Bad <call> tag]"
                    else:
                        # finish_reason == "length" and no tag
                        req['current_solution'] += small_model_output
                        if finish_reason == "length":
                            print(f"  [Req {req['id']}] --> INTERRUPTED (Length). Continuing...")
                        else:
                            req['status'] = 'done'
                            req['final_answer'] = req['current_solution']

        # --- Large Model Step ---
        large_model_batch = [req for req in request_pool if req["status"] == "waiting_for_large_model"]
        if large_model_batch:
            payloads = []
            for req in large_model_batch:
                prompt_body_for_lm = call_tag_pattern.sub(" ", req['current_solution']).strip()
                
                payloads.append({
                    "prompt": req['base_prompt'] + prompt_body_for_lm, 
                    "max_tokens": req['delegate_info']['max_tokens'],
                })

            request_ids = [req['id'] for req in large_model_batch]
            large_model_results = call_large_model_service_batch(payloads, request_ids, large_model_url)
            
            for req, result_obj in zip(large_model_batch, large_model_results):
                result_text = result_obj.get("text", "")
                finish_reason = result_obj.get("finish_reason", "error")

                if finish_reason == "error" or result_text.startswith("[Error:"):
                    req['status'] = 'failed'
                    req['final_answer'] = req['current_solution'] + "\n" + result_text
                else:
                    contribution_len = len(tokenizer.encode(result_text, add_special_tokens=False))
                    prompt_body_for_lm = call_tag_pattern.sub(" ", req['current_solution']).strip()
                    start_token = len(tokenizer.encode(prompt_body_for_lm, add_special_tokens=False))
                    end_token = start_token + contribution_len
                    
                    req['large_model_contributions'].append({
                        "start_token": start_token,
                        "end_token": end_token
                    })
                    
                    req['current_token_count'] += contribution_len
                    req['current_solution'] += result_text 

                    if finish_reason == "stop":
                        req['status'] = 'done'
                        req['final_answer'] = req['current_solution']
                        print(f"  [LM ID: {req['id']}] <-- Large model FINISHED.")
                    elif finish_reason == "length":
                        if req['current_token_count'] >= GLOBAL_MAX_TOKENS:
                            req['status'] = 'done'
                            req['final_answer'] = req['current_solution'] + f"\n[Info: Limit Reached]"
                        else:
                            req['status'] = 'waiting_for_small_model'
                            print(f"  [LM ID: {req['id']}] <-- Large model LENGTH limit. Back to small.")
                    else:
                        req['status'] = 'waiting_for_small_model'

        active_requests = sum(1 for req in request_pool if req["status"] not in ["done", "failed"])

    request_pool.sort(key=lambda r: r['id'])
    return request_pool


def _token_length(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _append_hsp_context(req: Dict[str, Any], text: str, tokenizer: AutoTokenizer, source: str) -> tuple[int, int]:
    start = _token_length(tokenizer, req["current_solution"])
    req["current_solution"] += text
    end = _token_length(tokenizer, req["current_solution"])
    req["current_token_count"] = end
    if source == "student":
        req["student_output_for_grading"] += text
    req["segments"].append({
        "source": source,
        "text": text,
        "loss": source == "student",
    })
    return start, end


def _build_teacher_prompt(req: Dict[str, Any], action: str) -> str:
    transcript = req["base_prompt"] + req["current_solution"]
    if action == "ask":
        instruction = (
            "\n\nYou are an external mathematical reasoning consultant. "
            "The student requested help at its current reasoning point. "
            "Provide one short, useful next step only. Do not produce control markers "
            "such as <ASK>, <VERIFY>, or <ACCEPT>, and do not take over the final answer.\n"
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


def _wrap_teacher_feedback(action: str, teacher_text: str) -> str:
    open_tag, close_tag = (
        ("<TEACHER_HELP>", "</TEACHER_HELP>")
        if action == "ask"
        else ("<TEACHER_REVIEW>", "</TEACHER_REVIEW>")
    )
    return f"\n{open_tag}\n{teacher_text.strip()}\n{close_tag}\n"


def _terminate_without_observation(req: Dict[str, Any], reason: str) -> None:
    event = req["events"][-1]
    event["observation_status"] = "omitted_no_context_budget"
    event["terminal_without_observation"] = True
    req["pending_action"] = None
    req["termination_reason"] = reason
    req["status"] = "done"
    req["final_answer"] = req["current_solution"]


def _resume_after_teacher_failure(req: Dict[str, Any], tokenizer: AutoTokenizer) -> None:
    notice = (
        "\n<ENVIRONMENT_NOTICE>External response unavailable. "
        "Complete the answer independently.</ENVIRONMENT_NOTICE>\n"
    )
    remaining = GLOBAL_MAX_TOKENS - req["current_token_count"]
    if _token_length(tokenizer, notice) > remaining:
        _terminate_without_observation(req, "Insufficient token budget for environment failure notice.")
        return
    req["pending_action"] = None
    _append_hsp_context(req, notice, tokenizer, source="environment")
    req["events"][-1]["observation_status"] = "delivered_environment_notice"
    req["status"] = "waiting_for_small_model"


def _fit_teacher_feedback(
    req: Dict[str, Any], action: str, teacher_text: str, tokenizer: AutoTokenizer
) -> tuple[str, bool]:
    def fits(candidate: str) -> bool:
        context = req["current_solution"] + _wrap_teacher_feedback(action, candidate)
        return _token_length(tokenizer, context) <= GLOBAL_MAX_TOKENS

    if fits(teacher_text):
        return teacher_text, False

    token_ids = tokenizer.encode(teacher_text, add_special_tokens=False)
    low, high = 0, len(token_ids)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = tokenizer.decode(token_ids[:middle], skip_special_tokens=True)
        if fits(candidate):
            low = middle
        else:
            high = middle - 1

    return tokenizer.decode(token_ids[:low], skip_special_tokens=True), True


def _record_review_decision(req: Dict[str, Any], student_text: str) -> None:
    accept_occurrences = student_text.count("<ACCEPT>")
    pending_idx = req.get("pending_review_event")
    if pending_idx is None:
        req["invalid_accept_count"] += accept_occurrences
        return

    event = req["events"][pending_idx]
    event["student_after_feedback"] = student_text
    event["accepted"] = accept_occurrences > 0
    if accept_occurrences > 0:
        req["accept_count"] += 1
        req["invalid_accept_count"] += max(accept_occurrences - 1, 0)
    req["pending_review_event"] = None


def _scored_answer_correctness(handler: Any, text: str, answer: str) -> bool | None:
    if handler.extract_answer(text) is None:
        return None
    return bool(handler.compare_answer(text, answer))


def _annotate_review_validity(req: Dict[str, Any], handler: Any, answer: str) -> None:
    for event in req["events"]:
        if event.get("action") != "verify" or event.get("error"):
            continue
        feedback = event.get("teacher_context_text") or ""
        tentative = event.get("student_before_feedback") or ""
        tentative_correct = _scored_answer_correctness(handler, tentative, answer)
        feedback_answer_correct = _scored_answer_correctness(handler, feedback, answer)
        proposed_answer = handler.extract_answer(feedback)
        final_response = req.get("student_output_for_grading", "")
        final_matches_feedback_answer = bool(
            proposed_answer is not None and handler.compare_answer(final_response, proposed_answer)
        )
        tentative_matches_feedback_answer = bool(
            proposed_answer is not None and handler.compare_answer(tentative, proposed_answer)
        )
        implicit_adoption_without_accept = bool(
            not event.get("accepted", False)
            and proposed_answer is not None
            and final_matches_feedback_answer
            and not tentative_matches_feedback_answer
        )
        feedback_is_correct = feedback_answer_correct
        if feedback_is_correct is None and tentative_correct is not None:
            lower_feedback = feedback.lower()
            if "verdict: correct" in lower_feedback:
                feedback_is_correct = tentative_correct
            elif "verdict: incorrect" in lower_feedback:
                feedback_is_correct = not tentative_correct
        event["tentative_answer_correct"] = tentative_correct
        event["feedback_answer_correct"] = feedback_answer_correct
        event["feedback_is_correct"] = feedback_is_correct
        event["final_matches_feedback_answer"] = final_matches_feedback_answer
        event["implicit_adoption_without_accept"] = implicit_adoption_without_accept
        event["tentative_answer_scope"] = event.get("student_before_feedback_scope")
        event["feedback_answer_scope"] = (
            "visible_teacher_context" if event.get("teacher_context_text") is not None else None
        )


def _queue_hsp_action(
    req: Dict[str, Any],
    action: str,
    ask_budget_tokens: int,
    verify_budget_tokens: int,
    student_before_feedback: str,
    forced: bool = False,
    student_requested_budget: int | None = None,
) -> None:
    # Use student-requested budget N from <ASK>N</ASK> / <VERIFY>N</VERIFY>,
    # falling back to the config default if not available.
    if student_requested_budget is not None and student_requested_budget > 0:
        budget = student_requested_budget
    else:
        budget = ask_budget_tokens if action == "ask" else verify_budget_tokens
    req["interaction_count"] += 1
    req[f"{action}_count"] += 1
    req["pending_action"] = action
    req["pending_requested_budget"] = budget
    req["events"].append({
        "action": action,
        "executed": True,
        "forced": forced,
        "requested_budget_tokens": budget,
        "student_before_feedback": student_before_feedback,
        "student_before_feedback_scope": "cumulative_student_visible",
        "teacher_text": None,
        "teacher_tokens_used": 0,
        "accepted": False if action == "verify" else None,
        "student_after_feedback": None,
        "observation_status": "pending",
        "error": None,
    })
    req["status"] = "waiting_for_teacher"


def run_hsp_generation(
    llm: vllm.LLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    large_model_url: str,
    max_interactions: int,
    ask_budget_tokens: int,
    verify_budget_tokens: int,
    student_temperature: float,
    teacher_help_temperature: float,
    teacher_review_temperature: float,
    collection_mode: str = "policy",
) -> List[Dict]:
    supported_modes = {"policy", "independent", "force_ask_first", "force_verify_after_draft"}
    if collection_mode not in supported_modes:
        raise ValueError(f"Unsupported HSP collection_mode: {collection_mode}")
    if max_interactions < 0:
        raise ValueError("max_interactions must not be negative.")
    if collection_mode in {"force_ask_first", "force_verify_after_draft"} and max_interactions < 1:
        raise ValueError(f"{collection_mode} requires max_interactions >= 1.")
    if ask_budget_tokens <= 0 or verify_budget_tokens <= 0:
        raise ValueError("ask_budget_tokens and verify_budget_tokens must be positive.")
    # Pre-compile patterns for <ASK>N</ASK> and <VERIFY>N</VERIFY>
    ask_tag_pattern = re.compile(r"<ASK>\s*(\d+)\s*</ASK>")
    verify_tag_pattern = re.compile(r"<VERIFY>\s*(\d+)\s*</VERIFY>")
    request_pool: List[Dict[str, Any]] = [{
        "id": i,
        "base_prompt": prompt,
        "current_solution": "",
        "student_output_for_grading": "",
        "status": "waiting_for_small_model",
        "allow_actions": collection_mode == "policy",
        "collection_mode": collection_mode,
        "force_verify_after_draft": collection_mode == "force_verify_after_draft",
        "turn_count": 0,
        "current_token_count": 0,
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
        "teacher_help_tokens": 0,
        "teacher_review_tokens": 0,
        "events": [],
        "segments": [],
        "large_model_contributions": [],
        "final_answer": None,
        "termination_reason": None,
    } for i, prompt in enumerate(prompts)]

    if collection_mode == "force_ask_first":
        for req in request_pool:
            forced_text = f"<ASK>{ask_budget_tokens}</ASK>"
            _append_hsp_context(req, forced_text, tokenizer, source="student")
            _queue_hsp_action(
                req,
                "ask",
                ask_budget_tokens,
                verify_budget_tokens,
                student_before_feedback=req["student_output_for_grading"],
                forced=True,
            )

    while any(req["status"] not in ("done", "failed") for req in request_pool):
        small_batch = [req for req in request_pool if req["status"] == "waiting_for_small_model"]
        if small_batch:
            current_prompts = []
            sampling_params = []
            active_batch = []
            for req in small_batch:
                remaining = GLOBAL_MAX_TOKENS - req["current_token_count"]
                if remaining <= 0:
                    req["status"] = "done"
                    req["final_answer"] = req["current_solution"]
                    continue

                params: Dict[str, Any] = {
                    "temperature": student_temperature,
                    "max_tokens": remaining,
                    "skip_special_tokens": False,
                }
                if req["allow_actions"]:
                    params["stop"] = ["</ASK>", "</VERIFY>"]
                    params["include_stop_str_in_output"] = True

                current_prompts.append(req["base_prompt"] + req["current_solution"])
                sampling_params.append(vllm.SamplingParams(**params))
                active_batch.append(req)

            if active_batch:
                outputs = llm.generate(current_prompts, sampling_params)
                for req, output in zip(active_batch, outputs):
                    completion = output.outputs[0]
                    student_text = completion.text
                    req["turn_count"] += 1
                    _append_hsp_context(req, student_text, tokenizer, source="student")
                    _record_review_decision(req, student_text)
                    invalid_protocol_count = _student_context_marker_count(student_text)
                    if invalid_protocol_count:
                        req["invalid_protocol_count"] += invalid_protocol_count
                        req["collection_error"] = "Student emitted a reserved environment marker."
                    if not req["allow_actions"]:
                        blocked_actions = (
                            len(ask_tag_pattern.findall(student_text))
                            + len(verify_tag_pattern.findall(student_text))
                        )
                        if blocked_actions:
                            req["denied_action_count"] += blocked_actions
                            if req["collection_mode"] != "policy":
                                req["collection_error"] = (
                                    "Controlled trajectory emitted a disabled interaction action."
                                )

                    if req["force_verify_after_draft"]:
                        req["force_verify_after_draft"] = False
                        if ask_tag_pattern.search(student_text) or verify_tag_pattern.search(student_text) or "<ACCEPT>" in student_text:
                            req["status"] = "done"
                            req["final_answer"] = req["current_solution"]
                            req["collection_error"] = "Initial draft contained an uncontrolled policy token."
                            continue
                        forced_text = f"\n<VERIFY>{verify_budget_tokens}</VERIFY>"
                        _append_hsp_context(req, forced_text, tokenizer, source="student")
                        _queue_hsp_action(
                            req,
                            "verify",
                            ask_budget_tokens,
                            verify_budget_tokens,
                            student_before_feedback=req["student_output_for_grading"],
                            forced=True,
                        )
                        continue

                    action = None
                    requested_budget = None
                    if req["allow_actions"]:
                        ask_match = ask_tag_pattern.search(student_text)
                        verify_match = verify_tag_pattern.search(student_text)
                        if ask_match:
                            action = "ask"
                            requested_budget = int(ask_match.group(1))
                        elif verify_match:
                            action = "verify"
                            requested_budget = int(verify_match.group(1))

                    if action is None:
                        req["status"] = "done"
                        req["final_answer"] = req["current_solution"]
                        continue

                    if req["interaction_count"] >= max_interactions:
                        req["denied_action_count"] += 1
                        limit_message = (
                            "\n<ENVIRONMENT_NOTICE>No further external interactions are available. "
                            "Complete the answer independently.</ENVIRONMENT_NOTICE>\n"
                        )
                        remaining = GLOBAL_MAX_TOKENS - req["current_token_count"]
                        if _token_length(tokenizer, limit_message) > remaining:
                            req["termination_reason"] = "Insufficient token budget for interaction-limit notice."
                            req["status"] = "done"
                            req["final_answer"] = req["current_solution"]
                            continue
                        _append_hsp_context(req, limit_message, tokenizer, source="environment")
                        req["allow_actions"] = False
                        req["status"] = "waiting_for_small_model"
                        continue

                    _queue_hsp_action(
                        req,
                        action,
                        ask_budget_tokens,
                        verify_budget_tokens,
                        student_before_feedback=req["student_output_for_grading"],
                        student_requested_budget=requested_budget,
                    )

        teacher_batch = [req for req in request_pool if req["status"] == "waiting_for_teacher"]
        if teacher_batch:
            payloads = []
            callable_teacher_batch = []
            for req in teacher_batch:
                action = req["pending_action"]
                is_help = action == "ask"
                remaining = GLOBAL_MAX_TOKENS - req["current_token_count"]
                empty_feedback_context = req["current_solution"] + _wrap_teacher_feedback(action, "")
                wrapper_tokens = max(_token_length(tokenizer, empty_feedback_context) - req["current_token_count"], 0)
                available_teacher_tokens = remaining - wrapper_tokens
                if available_teacher_tokens <= 0:
                    event = req["events"][-1]
                    event["error"] = "Teacher request skipped: insufficient total token budget for feedback context."
                    _terminate_without_observation(req, "Insufficient token budget for teacher observation.")
                    continue
                # Use student-requested budget from <ASK>N</ASK> / <VERIFY>N</VERIFY>
                student_budget = req.get("pending_requested_budget")
                if student_budget is not None and student_budget > 0:
                    base_budget = student_budget
                else:
                    base_budget = ask_budget_tokens if is_help else verify_budget_tokens
                teacher_budget = min(base_budget, available_teacher_tokens)
                req["events"][-1]["applied_budget_tokens"] = teacher_budget
                payloads.append({
                    "prompt": _build_teacher_prompt(req, req["pending_action"]),
                    "max_tokens": teacher_budget,
                    "temperature": teacher_help_temperature if is_help else teacher_review_temperature,
                    "top_p": 1.0,
                })
                callable_teacher_batch.append(req)
            if not callable_teacher_batch:
                continue
            results = call_large_model_service_batch(
                payloads, [req["id"] for req in callable_teacher_batch], large_model_url
            )
            for req, result in zip(callable_teacher_batch, results):
                action = req["pending_action"]
                event_idx = len(req["events"]) - 1
                event = req["events"][event_idx]
                teacher_text = result.get("text", "")
                finish_reason = result.get("finish_reason", "error")
                if finish_reason == "error" or teacher_text.startswith("[Error:"):
                    event["error"] = teacher_text
                    _resume_after_teacher_failure(req, tokenizer)
                    continue

                used_tokens = result.get("token_count", _token_length(tokenizer, teacher_text))
                reserved_marker = _reserved_hsp_marker(teacher_text)
                if reserved_marker is not None:
                    event["teacher_text"] = teacher_text
                    event["teacher_tokens_used"] = used_tokens
                    event["error"] = f"Teacher response contained reserved protocol marker: {reserved_marker}."
                    req["teacher_tokens_used"] += used_tokens
                    req[f"teacher_{'help' if action == 'ask' else 'review'}_tokens"] += used_tokens
                    _resume_after_teacher_failure(req, tokenizer)
                    continue

                if action == "ask" and _teacher_help_answer_leak(teacher_text):
                    event["teacher_text"] = teacher_text
                    event["teacher_tokens_used"] = used_tokens
                    event["error"] = "Teacher help response disclosed a final-answer candidate."
                    req["teacher_tokens_used"] += used_tokens
                    req["teacher_help_tokens"] += used_tokens
                    _resume_after_teacher_failure(req, tokenizer)
                    continue

                context_teacher_text, feedback_truncated = _fit_teacher_feedback(req, action, teacher_text, tokenizer)
                wrapped_feedback = _wrap_teacher_feedback(action, context_teacher_text)
                start, end = _append_hsp_context(req, wrapped_feedback, tokenizer, source="teacher")
                context_teacher_tokens = _token_length(tokenizer, context_teacher_text)
                req["teacher_tokens_used"] += used_tokens
                req[f"teacher_{'help' if action == 'ask' else 'review'}_tokens"] += used_tokens
                req["large_model_contributions"].append({
                    "start_token": start,
                    "end_token": end,
                    "action": action,
                    "teacher_tokens_used": used_tokens,
                    "teacher_context_tokens": context_teacher_tokens,
                    "feedback_truncated": feedback_truncated,
                })
                event["teacher_text"] = teacher_text
                event["teacher_tokens_used"] = used_tokens
                event["teacher_context_text"] = context_teacher_text
                event["teacher_context_tokens"] = context_teacher_tokens
                event["feedback_truncated"] = feedback_truncated
                event["observation_status"] = "delivered_teacher_observation"
                if action == "verify":
                    req["pending_review_event"] = event_idx

                req["pending_action"] = None
                req["status"] = "waiting_for_small_model"

    request_pool.sort(key=lambda r: r["id"])
    return request_pool


def main(args):
    _validate_data_role(args.dataset, args.data_role)
    final_output_path = _result_output_path(args)
    if os.path.exists(final_output_path) and not args.overwrite_results:
        raise FileExistsError(
            f"Results file already exists: {final_output_path}. "
            "Use --output_tag for a new collection run or --overwrite_results explicitly."
        )
    print(f"STORAGE_PATH: {STORAGE_PATH}")
    print(f"Small model: {args.small_model}, Dataset: {args.dataset}")

    tokenizer_path = args.tokenizer_path or args.small_model
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    model = vllm.LLM(
        model=args.small_model,
        tokenizer=tokenizer_path,
        gpu_memory_utilization=0.85,
        trust_remote_code=True
    )
    
    handler = datasets_loader.get_dataset_handler(args.dataset, args.name)
    base_questions, base_answers = handler.load_data()
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("--max_examples must be positive when supplied.")
        base_questions = base_questions[: args.max_examples]
        base_answers = base_answers[: args.max_examples]
    if args.samples_per_question <= 0:
        raise ValueError("--samples_per_question must be positive.")
    questions = []
    answers = []
    sample_indices = []
    for question, answer in zip(base_questions, base_answers):
        for sample_index in range(args.samples_per_question):
            questions.append(question)
            answers.append(answer)
            sample_indices.append(sample_index)

    prompts = []
    for question in questions:
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": _user_prompt_content(question)}]
            # enable_thinking=False 只对特定模型有效，加个 try-except 或移除
            try:
                p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(p)
        else:
            # Fallback
            prompts.append("User: " + _user_prompt_content(question) + "\nAssistant:")

    time_start = time.time()
    print("\nStarting collaborative generation process...")
    if args.interaction_policy == "hsp":
        request_pool = run_hsp_generation(
            model,
            tokenizer,
            prompts,
            args.large_model_url,
            max_interactions=args.max_interactions,
            ask_budget_tokens=args.ask_budget_tokens,
            verify_budget_tokens=args.verify_budget_tokens,
            student_temperature=args.student_temperature,
            teacher_help_temperature=args.teacher_help_temperature,
            teacher_review_temperature=args.teacher_review_temperature,
            collection_mode=args.collection_mode,
        )
    else:
        if args.collection_mode != "policy":
            raise ValueError("--collection_mode is only supported with --interaction_policy hsp.")
        request_pool = run_relay_generation(model, tokenizer, prompts, args.large_model_url)
    print("Collaborative generation finished.\n")
    time_end = time.time()

    responses = [r['final_answer'] if r['final_answer'] is not None else r['current_solution'] for r in request_pool]
    all_contributions = [r['large_model_contributions'] for r in request_pool]
    grading_responses = (
        [r["student_output_for_grading"] for r in request_pool]
        if args.interaction_policy == "hsp"
        else responses
    )

    scores, average_score = handler.get_score(grading_responses, answers)

    results = []
    larger_model_tokens = 0
    total_tokens = 0

    for i, (q, a, sample_index, r, student_r, s, contribs, req) in enumerate(
        zip(questions, answers, sample_indices, responses, grading_responses, scores, all_contributions, request_pool)
    ):
        if args.interaction_policy == "hsp":
            _annotate_review_validity(req, handler, a)
        result_item = {
            "question": q, 
            "answer": a, 
            "sample_index": sample_index,
            "data_role": args.data_role,
            "dataset_name": args.name,
            "output_tag": args.output_tag,
            "response": r, 
            "student_response_for_grading": student_r,
            "score": s, 
            "large_model_contributions": contribs,
            "interaction_policy": args.interaction_policy,
        }
        if args.interaction_policy == "hsp":
            result_item.update({
                "events": req["events"],
                "ask_count": req["ask_count"],
                "verify_count": req["verify_count"],
                "accept_count": req["accept_count"],
                "invalid_accept_count": req["invalid_accept_count"],
                "invalid_protocol_count": req["invalid_protocol_count"],
                "denied_action_count": req["denied_action_count"],
                "teacher_tokens_used": req["teacher_tokens_used"],
                "collection_mode": args.collection_mode,
                "collection_error": req.get("collection_error"),
                "termination_reason": req.get("termination_reason"),
                "segments": [
                    {"source": "user", "text": _user_prompt_content(q), "loss": False},
                    *req["segments"],
                ],
            })
        results.append(result_item)
        
        if args.interaction_policy == "hsp":
            larger_model_tokens += req["teacher_tokens_used"]
        elif contribs:
            for c in contribs:
                larger_model_tokens += (c['end_token'] - c['start_token'])
        
        if r:
            total_tokens += len(tokenizer.encode(r, add_special_tokens=False))

    print(f"Average score: {average_score}")

    output_dir = os.path.dirname(final_output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {final_output_path}")

    summary_output_path = _summary_output_path(final_output_path)
    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump({
            'model': args.small_model,
            'dataset': args.dataset,
            'dataset_name': args.name,
            'output_tag': args.output_tag,
            'data_role': args.data_role,
            'score': round(average_score*100, 2),
            'larger_model': args.larger_model,
            'interaction_policy': args.interaction_policy,
            'collection_mode': args.collection_mode if args.interaction_policy == "hsp" else None,
            'max_examples': args.max_examples,
            'samples_per_question': args.samples_per_question,
            'real_call_tokens': larger_model_tokens,
            'total_tokens': total_tokens,
            'real_call_tokens_ratio': (larger_model_tokens / total_tokens) if total_tokens > 0 else 0,
            'time': time_end - time_start,
            'per_item_results': final_output_path,
        }, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_output_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_model", type=str, default="../LLaMA-Factory/saves/qwen3-0.6b-base/full/sft/checkpoint-500")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--large_model_url", type=str, default="http://127.0.0.1:7778/generate")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--data_role", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--larger_model", type=str, default='7B')
    parser.add_argument("--interaction_policy", choices=["relay_call", "hsp"], default="relay_call")
    parser.add_argument("--max_interactions", type=int, default=3)
    parser.add_argument("--ask_budget_tokens", type=int, default=64)
    parser.add_argument("--verify_budget_tokens", type=int, default=96)
    parser.add_argument("--student_temperature", type=float, default=0.7)
    parser.add_argument("--teacher_help_temperature", type=float, default=0.7)
    parser.add_argument("--teacher_review_temperature", type=float, default=0.0)
    parser.add_argument("--max_examples", type=int, default=None, help="Limit dataset questions for smoke tests.")
    parser.add_argument("--samples_per_question", type=int, default=1)
    parser.add_argument("--output_tag", type=str, default=None)
    parser.add_argument("--overwrite_results", action="store_true")
    parser.add_argument(
        "--collection_mode",
        choices=["policy", "independent", "force_ask_first", "force_verify_after_draft"],
        default="policy",
    )
    args = parser.parse_args()
    main(args)
