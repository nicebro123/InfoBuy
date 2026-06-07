"""Validate HSP data, tokenizer tokens, and initial RL configuration.

This check is intentionally lightweight for dataset inspection. Tokenizer and
OmegaConf dependencies are imported only when their corresponding checks are
requested.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from .hsp_collator import CONTEXT_MARKER_TOKENS, POLICY_ACTION_TOKENS
except ImportError:
    from hsp_collator import CONTEXT_MARKER_TOKENS, POLICY_ACTION_TOKENS


EXPECTED_SAMPLE_TYPES = {
    "normal",
    "ask_help",
    "verify_confirm",
    "verify_accept_correction",
    "verify_reject_bad_feedback",
    "verify_uncertain",
}
ALL_RESERVED_TOKENS = tuple(POLICY_ACTION_TOKENS) + tuple(CONTEXT_MARKER_TOKENS)
HSP_REWARD_WEIGHT_KEYS = (
    "teacher_token_budget",
    "teacher_cost_weight",
    "useful_accept_weight",
    "resist_bad_review_weight",
    "wrong_accept_weight",
    "wrong_reject_weight",
    "implicit_adoption_weight",
    "wrong_implicit_adoption_weight",
    "unsupported_accept_weight",
    "invalid_accept_weight",
    "invalid_protocol_weight",
    "denied_action_weight",
    "teacher_error_weight",
    "independent_correct_weight",
)


def validate_hsp_reward_profile(
    reward: dict[str, Any],
    profile_name: str,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if reward.get("reward_type") != "batch":
        errors.append(f"HSP {profile_name} requires reward_type=batch.")
    if "math_hsp_group.py" not in str(reward.get("reward_function", "")):
        warnings.append(f"{profile_name} is not the provided HSP outcome-cost-trust reward.")

    reward_kwargs = reward.get("reward_function_kwargs", {})
    missing_reward_weights = [key for key in HSP_REWARD_WEIGHT_KEYS if key not in reward_kwargs]
    if missing_reward_weights:
        errors.append(
            f"HSP {profile_name} must explicitly configure all active or telemetry weights: "
            + ", ".join(missing_reward_weights)
            + "."
        )
        return errors, warnings

    if float(reward_kwargs["teacher_token_budget"]) <= 0:
        errors.append(f"HSP {profile_name} teacher_token_budget must be positive.")
    negative_weights = [
        key
        for key in HSP_REWARD_WEIGHT_KEYS
        if key != "teacher_token_budget" and float(reward_kwargs[key]) < 0
    ]
    if negative_weights:
        errors.append(f"HSP {profile_name} weights must not be negative: " + ", ".join(negative_weights) + ".")
    if float(reward_kwargs["teacher_error_weight"]) == 0:
        warnings.append(f"{profile_name} records failed teacher calls without student reward penalties.")
    return errors, warnings


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    return any(token in text for token in tokens)


def _contains_exact_wrapper(text: str, open_token: str, close_token: str) -> bool:
    return (
        text.count(open_token) == 1
        and text.count(close_token) == 1
        and text.find(open_token) < text.find(close_token)
    )


def load_examples(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    with source.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("train", "data", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("HSP dataset must be JSONL, a JSON list, or contain train/data/examples.")


def validate_dataset(examples: Iterable[dict[str, Any]], require_all_types: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    type_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    teacher_segment_count = 0
    trainable_student_segment_count = 0
    checked_examples = 0

    for index, example in enumerate(examples):
        checked_examples += 1
        prefix = f"example[{index}]"
        segments = example.get("segments")
        if not isinstance(segments, list) or not segments:
            errors.append(f"{prefix}: segments must be a non-empty list.")
            continue
        sample_type = example.get("sample_type")
        if sample_type:
            type_counts[str(sample_type)] += 1

        pending_review_context = False
        expected_observation: str | None = None
        awaiting_student_continuation = False
        trainable_text = ""
        for segment_index, segment in enumerate(segments):
            segment_prefix = f"{prefix}.segments[{segment_index}]"
            source = segment.get("source")
            text = str(segment.get("text", ""))
            has_loss = bool(segment.get("loss", False))
            if source not in {"system", "user", "student", "teacher", "environment"}:
                errors.append(f"{segment_prefix}: unsupported source={source!r}.")
            if has_loss and source != "student":
                errors.append(f"{segment_prefix}: only student segments may have loss=true.")

            if source in {"system", "user"} and _contains_any(text, ALL_RESERVED_TOKENS):
                errors.append(f"{segment_prefix}: prompt context contains a reserved HSP marker.")
            if source in {"system", "user"} and (expected_observation is not None or pending_review_context):
                errors.append(f"{segment_prefix}: prompt context interrupts an unresolved interaction.")

            if source == "teacher":
                teacher_segment_count += 1
                if _contains_any(text, POLICY_ACTION_TOKENS):
                    errors.append(f"{segment_prefix}: teacher observation contains a policy action token.")
                if expected_observation is None:
                    errors.append(f"{segment_prefix}: unsolicited teacher observation without a preceding request.")
                expected_wrapper = (
                    ("<TEACHER_HELP>", "</TEACHER_HELP>")
                    if expected_observation == "ask"
                    else ("<TEACHER_REVIEW>", "</TEACHER_REVIEW>")
                )
                if expected_observation is not None and not _contains_exact_wrapper(text, *expected_wrapper):
                    errors.append(f"{segment_prefix}: teacher observation does not match the pending request.")
                unexpected_tokens = set(CONTEXT_MARKER_TOKENS).difference(expected_wrapper)
                if _contains_any(text, unexpected_tokens):
                    errors.append(f"{segment_prefix}: teacher observation contains mismatched environment markers.")
                if expected_observation == "verify" and _contains_exact_wrapper(text, *expected_wrapper):
                    pending_review_context = True
                if expected_observation is not None and _contains_exact_wrapper(text, *expected_wrapper):
                    awaiting_student_continuation = True
                expected_observation = None

            if source == "environment":
                teacher_tokens = ("<TEACHER_HELP>", "</TEACHER_HELP>", "<TEACHER_REVIEW>", "</TEACHER_REVIEW>")
                if _contains_any(text, POLICY_ACTION_TOKENS) or _contains_any(text, teacher_tokens):
                    errors.append(f"{segment_prefix}: environment notice contains a policy or teacher marker.")
                if not _contains_exact_wrapper(text, "<ENVIRONMENT_NOTICE>", "</ENVIRONMENT_NOTICE>"):
                    errors.append(f"{segment_prefix}: invalid environment notice wrapper.")
                if expected_observation is None:
                    errors.append(f"{segment_prefix}: unsolicited environment notice without a preceding request.")
                elif _contains_exact_wrapper(text, "<ENVIRONMENT_NOTICE>", "</ENVIRONMENT_NOTICE>"):
                    awaiting_student_continuation = True
                expected_observation = None

            if source == "student":
                if _contains_any(text, CONTEXT_MARKER_TOKENS):
                    errors.append(f"{segment_prefix}: student output contains a reserved environment marker.")
                if expected_observation is not None:
                    errors.append(f"{segment_prefix}: student continuation appears before the requested observation.")

                ask_request_count = len(re.findall(r"<ASK>\s*\d+\s*</ASK>", text))
                verify_request_count = len(re.findall(r"<VERIFY>\s*\d+\s*</VERIFY>", text))
                request_count = ask_request_count + verify_request_count
                if request_count > 1:
                    errors.append(f"{segment_prefix}: at most one help request may appear in a student segment.")
                elif request_count == 1 and re.search(r"(?:<ASK>\s*\d+\s*</ASK>|<VERIFY>\s*\d+\s*</VERIFY>)\s*$", text) is None:
                    errors.append(f"{segment_prefix}: a help request token must terminate its student segment.")

                accept_count = text.count("<ACCEPT>")
                if accept_count > 1:
                    errors.append(f"{segment_prefix}: at most one <ACCEPT> may appear in a student segment.")
                if accept_count and not pending_review_context:
                    errors.append(f"{segment_prefix}: <ACCEPT> appears without a pending teacher review.")

                if has_loss:
                    trainable_student_segment_count += 1
                    trainable_text += text
                if request_count == 1:
                    expected_observation = "ask" if ask_request_count else "verify"
                awaiting_student_continuation = False

            if source == "student" and pending_review_context:
                pending_review_context = False

        if expected_observation is not None:
            errors.append(f"{prefix}: requested {expected_observation} observation is missing.")
        if awaiting_student_continuation:
            errors.append(f"{prefix}: observation is missing a student continuation.")
        if not trainable_text:
            errors.append(f"{prefix}: no trainable student text.")
            continue
        for action_token in POLICY_ACTION_TOKENS:
            action_counts[action_token] += trainable_text.count(action_token)

    if checked_examples == 0:
        errors.append("Dataset has no examples.")
    missing_types = sorted(EXPECTED_SAMPLE_TYPES.difference(type_counts))
    if missing_types:
        message = "Dataset does not contain protocol types: " + ", ".join(missing_types) + "."
        if require_all_types:
            errors.append(message)
        else:
            warnings.append(message)
    for token in POLICY_ACTION_TOKENS:
        if action_counts[token] == 0:
            warnings.append(f"No trainable occurrence of {token} was found.")

    return {
        "checked_examples": checked_examples,
        "sample_type_counts": dict(sorted(type_counts.items())),
        "policy_action_counts": dict(action_counts),
        "teacher_segment_count": teacher_segment_count,
        "trainable_student_segment_count": trainable_student_segment_count,
        "errors": errors,
        "warnings": warnings,
    }


def validate_tokenizer(model_path: str, require_context_tokens: bool = False) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Tokenizer validation requires transformers to be installed.") from error

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    errors: list[str] = []
    warnings: list[str] = []
    tokens_to_check = list(POLICY_ACTION_TOKENS)
    if require_context_tokens:
        tokens_to_check.extend(CONTEXT_MARKER_TOKENS)
    token_ids: dict[str, list[int]] = {}

    if not getattr(tokenizer, "is_fast", False):
        errors.append("Tokenizer is not fast; exact SFT offset masking cannot be reproduced.")

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in tokens_to_check:
        ids = tokenizer.encode(token, add_special_tokens=False)
        token_ids[token] = ids
        if len(ids) != 1:
            errors.append(f"{token} is not a single tokenizer token: {ids}.")
        elif unk_token_id is not None and ids[0] == unk_token_id:
            errors.append(f"{token} maps to unk_token_id.")

    if not require_context_tokens:
        warnings.append("Context wrapper atomic-token validation was skipped.")

    return {
        "model_path": model_path,
        "checked_tokens": token_ids,
        "errors": errors,
        "warnings": warnings,
    }


def validate_rl_config(config: dict[str, Any], model_override: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    data = config.get("data", {})
    algorithm = config.get("algorithm", {})
    worker = config.get("worker", {})
    rollout = worker.get("rollout", {})
    reward = worker.get("reward", {})
    val_reward = worker.get("val_reward")

    if rollout.get("interaction_policy") != "hsp":
        errors.append("worker.rollout.interaction_policy must be hsp.")
    if algorithm.get("adv_estimator") != "grpo":
        errors.append("The validated HSP path currently requires algorithm.adv_estimator=grpo.")
    reward_errors, reward_warnings = validate_hsp_reward_profile(reward, "worker.reward")
    errors.extend(reward_errors)
    warnings.extend(reward_warnings)
    if isinstance(val_reward, dict) and val_reward:
        val_reward_errors, val_reward_warnings = validate_hsp_reward_profile(val_reward, "worker.val_reward")
        errors.extend(val_reward_errors)
        warnings.extend(val_reward_warnings)

    max_response_length = int(data.get("max_response_length", 0))
    max_prompt_length = int(data.get("max_prompt_length", 0))
    global_max_tokens = int(rollout.get("global_max_tokens", max_response_length))
    max_num_batched_tokens = int(rollout.get("max_num_batched_tokens", 0))
    if global_max_tokens > max_response_length:
        errors.append("global_max_tokens must not exceed data.max_response_length.")
    if max_num_batched_tokens < max_prompt_length + max_response_length:
        errors.append("max_num_batched_tokens must cover max_prompt_length + max_response_length.")
    for key in ("max_interactions", "ask_budget_tokens", "verify_budget_tokens"):
        if int(rollout.get(key, 0)) <= 0:
            errors.append(f"worker.rollout.{key} must be positive.")
    if rollout.get("val_override_config", {}).get("skip_special_tokens", False):
        errors.append("HSP rollout validation must not set val_override_config.skip_special_tokens=true.")

    action_values = {
        "ask_token": "<ASK>",
        "end_ask_token": "</ASK>",
        "verify_token": "<VERIFY>",
        "end_verify_token": "</VERIFY>",
        "accept_token": "<ACCEPT>",
    }
    for key, expected in action_values.items():
        if rollout.get(key, expected) != expected:
            warnings.append(f"worker.rollout.{key} differs from the SFT preflight token {expected}.")

    configured_model = worker.get("actor", {}).get("model", {}).get("model_path", "")
    if not model_override and str(configured_model).startswith("/path/to/"):
        errors.append("worker.actor.model.model_path is still a placeholder and no --model_path override was supplied.")

    return {"errors": errors, "warnings": warnings}


def load_rl_config(path: str) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("RL config validation requires omegaconf or pyyaml to be installed.") from error
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        return data or {}
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def validate_sft_rl_length_contract(model_path: str, config: dict[str, Any]) -> dict[str, Any]:
    contract_path = Path(model_path) / "hsp_training_contract.json"
    errors: list[str] = []
    warnings: list[str] = []
    if not contract_path.exists():
        errors.append(
            "SFT checkpoint does not contain hsp_training_contract.json; "
            "the SFT/RL context-length compatibility cannot be verified."
        )
        return {"contract_path": str(contract_path), "errors": errors, "warnings": warnings}

    with contract_path.open("r", encoding="utf-8") as stream:
        contract = json.load(stream)
    max_seq_length = int(contract.get("max_seq_length", 0))
    data = config.get("data", {})
    rollout = config.get("worker", {}).get("rollout", {})
    required_length = int(data.get("max_prompt_length", 0)) + int(
        rollout.get("global_max_tokens", data.get("max_response_length", 0))
    )
    if max_seq_length < required_length:
        errors.append(
            f"SFT max_seq_length={max_seq_length} is shorter than the RL visible sequence "
            f"budget={required_length}; retrain with a sufficient length or reduce RL limits."
        )
    return {
        "contract_path": str(contract_path),
        "sft_max_seq_length": max_seq_length,
        "rl_required_visible_length": required_length,
        "errors": errors,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HSP experiment inputs before SFT or GRPO.")
    parser.add_argument("--dataset", help="Structured HSP JSON/JSONL dataset to inspect.")
    parser.add_argument("--model_path", help="SFT checkpoint/tokenizer to validate for HSP action tokens.")
    parser.add_argument("--rl_config", help="HSP GRPO YAML config to validate.")
    parser.add_argument("--require_all_types", action="store_true")
    parser.add_argument("--require_context_tokens", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not any((args.dataset, args.model_path, args.rl_config)):
        raise ValueError("Specify at least one of --dataset, --model_path, or --rl_config.")

    report: dict[str, Any] = {}
    if args.dataset:
        report["dataset"] = validate_dataset(load_examples(args.dataset), args.require_all_types)
    if args.model_path:
        report["tokenizer"] = validate_tokenizer(args.model_path, args.require_context_tokens)
    rl_config = None
    if args.rl_config:
        rl_config = load_rl_config(args.rl_config)
        report["rl_config"] = validate_rl_config(rl_config, args.model_path)
    if args.model_path and rl_config is not None:
        report["length_contract"] = validate_sft_rl_length_contract(args.model_path, rl_config)

    errors = [error for result in report.values() for error in result.get("errors", [])]
    if args.as_json:
        report["ok"] = not errors
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        for section, result in report.items():
            print(f"[{section}]")
            for key, value in result.items():
                if key not in {"errors", "warnings"}:
                    print(f"{key}: {value}")
            for warning in result.get("warnings", []):
                print(f"WARNING: {warning}")
            for error in result.get("errors", []):
                print(f"ERROR: {error}")
    if errors:
        raise SystemExit(f"HSP preflight failed with {len(errors)} error(s).")
    if not args.as_json:
        print("HSP preflight passed.")


if __name__ == "__main__":
    main()
