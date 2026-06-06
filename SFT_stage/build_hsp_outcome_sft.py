"""Select successful, low-cost HSP rollout traces for replay SFT.

The cold-start builder teaches protocol syntax. This builder is for the next
iteration: collect diverse HSP evaluation traces, select the best measured
trajectory per problem, then replay only student spans as SFT targets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .hsp_collator import CONTEXT_MARKER_TOKENS
    from .preflight_hsp import validate_dataset
except ImportError:
    from hsp_collator import CONTEXT_MARKER_TOKENS
    from preflight_hsp import validate_dataset


def load_result_items(paths: list[str]) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list of HSP evaluation items.")
        records.extend((path, item) for item in data)
    return records


def utility(
    item: dict[str, Any],
    teacher_token_budget: float,
    teacher_cost_weight: float,
    invalid_accept_weight: float,
    denied_action_weight: float,
) -> float:
    if teacher_token_budget <= 0:
        raise ValueError("teacher_token_budget must be positive.")
    score = float(item.get("score", 0.0))
    teacher_cost = float(item.get("teacher_tokens_used", 0)) / teacher_token_budget
    return (
        score
        - teacher_cost_weight * teacher_cost
        - invalid_accept_weight * int(item.get("invalid_accept_count", 0))
        - denied_action_weight * int(item.get("denied_action_count", 0))
    )


def validate_replay_candidate(
    item: dict[str, Any],
    min_score: float,
    max_teacher_tokens: int | None,
    include_event_errors: bool,
    include_invalid_actions: bool,
    require_train_data: bool,
) -> str | None:
    if item.get("interaction_policy") != "hsp":
        return "non_hsp"
    if require_train_data and item.get("data_role") != "train":
        return "non_train_provenance"
    if item.get("collection_error"):
        return "collection_error"
    if int(item.get("invalid_protocol_count", 0)) > 0:
        return "forged_environment_marker"
    if float(item.get("score", 0.0)) < min_score:
        return "below_min_score"
    if max_teacher_tokens is not None and int(item.get("teacher_tokens_used", 0)) > max_teacher_tokens:
        return "above_teacher_token_limit"
    if not include_invalid_actions and (
        int(item.get("invalid_accept_count", 0)) > 0 or int(item.get("denied_action_count", 0)) > 0
    ):
        return "invalid_action"
    if not include_event_errors and any(event.get("error") for event in item.get("events", [])):
        return "event_error"
    events = item.get("events", [])
    if any(bool(event.get("implicit_adoption_without_accept", False)) for event in events):
        return "implicit_adoption_without_accept"
    accepted_events = [
        event for event in events
        if event.get("action") == "verify" and bool(event.get("accepted", False))
    ]
    if len(accepted_events) != int(item.get("accept_count", 0)):
        return "accept_event_mismatch"
    for event in accepted_events:
        if event.get("feedback_answer_correct") is False:
            return "wrong_accept"
        if (
            event.get("tentative_answer_scope") != "cumulative_student_visible"
            or event.get("feedback_answer_scope") != "visible_teacher_context"
        ):
            return "unvalidated_accept"
        if event.get("feedback_answer_correct") is not True:
            return "unvalidated_accept"
        if event.get("tentative_answer_correct") is True:
            return "redundant_accept"

    segments = item.get("segments")
    if not isinstance(segments, list) or not segments:
        return "missing_segments"
    if segments[0].get("source") not in {"system", "user"}:
        return "missing_initial_context"
    if not any(segment.get("source") == "student" and segment.get("loss", False) for segment in segments):
        return "missing_student_target"
    trainable_text = ""
    for segment in segments:
        source = segment.get("source")
        text = str(segment.get("text", ""))
        has_loss = bool(segment.get("loss", False))
        if has_loss and source != "student":
            return "non_student_loss"
        if source == "student" and any(token in text for token in CONTEXT_MARKER_TOKENS):
            return "forged_environment_marker"
        if has_loss and source == "student":
            trainable_text += text
    protocol_report = validate_dataset([{"segments": segments}])
    if protocol_report["errors"]:
        return "invalid_protocol_segments"
    event_action_counts = {
        "<ASK>": int(item.get("ask_count", 0)),
        "<VERIFY>": int(item.get("verify_count", 0)),
        "<ACCEPT>": int(item.get("accept_count", 0)),
    }
    for token, expected_count in event_action_counts.items():
        if trainable_text.count(token) != expected_count:
            return "action_event_mismatch"
    return None


def trajectory_type(item: dict[str, Any]) -> str:
    asks = int(item.get("ask_count", 0))
    verifies = int(item.get("verify_count", 0))
    accepts = int(item.get("accept_count", 0))
    if asks == 0 and verifies == 0:
        return "independent_success"
    if asks > 0 and verifies > 0:
        return "mixed_interaction_success"
    if asks > 0:
        return "ask_success"
    if accepts > 0:
        return "verify_accept_success"
    return "verify_without_accept_success"


def build_replay_examples(
    records: list[tuple[str, dict[str, Any]]],
    teacher_token_budget: float = 192.0,
    teacher_cost_weight: float = 0.15,
    invalid_accept_weight: float = 0.10,
    denied_action_weight: float = 0.05,
    min_score: float = 1.0,
    max_teacher_tokens: int | None = None,
    keep_per_problem: int = 1,
    include_event_errors: bool = False,
    include_invalid_actions: bool = False,
    require_train_data: bool = True,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if keep_per_problem <= 0:
        raise ValueError("keep_per_problem must be positive.")
    grouped: dict[str, list[tuple[str, dict[str, Any], float]]] = defaultdict(list)
    skipped: Counter[str] = Counter()

    for source_path, item in records:
        reason = validate_replay_candidate(
            item,
            min_score=min_score,
            max_teacher_tokens=max_teacher_tokens,
            include_event_errors=include_event_errors,
            include_invalid_actions=include_invalid_actions,
            require_train_data=require_train_data,
        )
        if reason is not None:
            skipped[reason] += 1
            continue
        problem_key = str(item.get("question") or item.get("source_id") or item.get("id") or "")
        if not problem_key:
            skipped["missing_problem_key"] += 1
            continue
        measured_utility = utility(
            item,
            teacher_token_budget=teacher_token_budget,
            teacher_cost_weight=teacher_cost_weight,
            invalid_accept_weight=invalid_accept_weight,
            denied_action_weight=denied_action_weight,
        )
        grouped[problem_key].append((source_path, item, measured_utility))

    selected: list[dict[str, Any]] = []
    for problem_index, problem_key in enumerate(sorted(grouped)):
        candidates = sorted(
            grouped[problem_key],
            key=lambda value: (
                -value[2],
                int(value[1].get("teacher_tokens_used", 0)),
                int(value[1].get("ask_count", 0)) + int(value[1].get("verify_count", 0)),
            ),
        )
        for selection_index, (source_path, item, measured_utility) in enumerate(candidates[:keep_per_problem]):
            selected.append({
                "id": f"outcome_{problem_index:07d}_{selection_index}",
                "source_id": str(item.get("question", problem_key)),
                "trajectory_type": trajectory_type(item),
                "generation_mode": "outcome_selected_rollout",
                "source_result_file": source_path,
                "collection_mode": item.get("collection_mode", "policy"),
                "data_role": item.get("data_role"),
                "gold_answer": item.get("answer"),
                "score": float(item.get("score", 0.0)),
                "utility": measured_utility,
                "teacher_tokens_used": int(item.get("teacher_tokens_used", 0)),
                "ask_count": int(item.get("ask_count", 0)),
                "verify_count": int(item.get("verify_count", 0)),
                "accept_count": int(item.get("accept_count", 0)),
                "events": item.get("events", []),
                "segments": item["segments"],
            })
    return selected, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build outcome-selected HSP replay SFT JSONL.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more results_*_hsp.json files.")
    parser.add_argument("--output", required=True, help="Structured replay SFT JSONL output.")
    parser.add_argument("--teacher_token_budget", type=float, default=192.0)
    parser.add_argument("--teacher_cost_weight", type=float, default=0.15)
    parser.add_argument("--invalid_accept_weight", type=float, default=0.10)
    parser.add_argument("--denied_action_weight", type=float, default=0.05)
    parser.add_argument("--min_score", type=float, default=1.0)
    parser.add_argument("--max_teacher_tokens", type=int, default=None)
    parser.add_argument("--keep_per_problem", type=int, default=1)
    parser.add_argument("--include_event_errors", action="store_true")
    parser.add_argument("--include_invalid_actions", action="store_true")
    parser.add_argument(
        "--allow_non_train_inputs",
        action="store_true",
        help="Debug only: accept traces not marked data_role=train; do not use for reportable training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected, skipped = build_replay_examples(
        load_result_items(args.input),
        teacher_token_budget=args.teacher_token_budget,
        teacher_cost_weight=args.teacher_cost_weight,
        invalid_accept_weight=args.invalid_accept_weight,
        denied_action_weight=args.denied_action_weight,
        min_score=args.min_score,
        max_teacher_tokens=args.max_teacher_tokens,
        keep_per_problem=args.keep_per_problem,
        include_event_errors=args.include_event_errors,
        include_invalid_actions=args.include_invalid_actions,
        require_train_data=not args.allow_non_train_inputs,
    )
    if not selected:
        skipped_text = json.dumps(dict(sorted(skipped.items())), sort_keys=True)
        raise ValueError(
            "No qualifying HSP rollout traces were selected for replay SFT. "
            f"Skipped candidates: {skipped_text}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for example in selected:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")

    type_counts = Counter(example["trajectory_type"] for example in selected)
    print(f"Wrote {len(selected)} outcome-selected HSP examples to {output_path}.")
    print("Selected trajectory types: " + json.dumps(dict(sorted(type_counts.items())), sort_keys=True))
    print("Skipped candidates: " + json.dumps(dict(sorted(skipped.items())), sort_keys=True))
    print("Selection utility uses measured score minus teacher/action costs; it requires diverse rollout candidates.")


if __name__ == "__main__":
    main()
