"""Summarize HSP evaluation traces for action calibration analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _subset_metrics(items: list[dict[str, Any]], predicate) -> dict[str, float | int | None]:
    subset = [item for item in items if predicate(item)]
    return {
        "count": len(subset),
        "fraction": len(subset) / len(items) if items else 0.0,
        "mean_score": _mean(float(item.get("score", 0.0)) for item in subset),
        "mean_teacher_tokens": _mean(float(item.get("teacher_tokens_used", 0)) for item in subset),
    }


def summarize_items(items: list[dict[str, Any]], source: str | None = None) -> dict[str, Any]:
    hsp_items = [item for item in items if item.get("interaction_policy") == "hsp"]
    if not hsp_items:
        raise ValueError("No HSP result items were found.")

    events = [event for item in hsp_items for event in item.get("events", [])]
    teacher_tokens = [float(item.get("teacher_tokens_used", 0)) for item in hsp_items]
    action_keys = (
        "ask_count",
        "verify_count",
        "accept_count",
        "invalid_accept_count",
        "invalid_protocol_count",
        "denied_action_count",
    )
    action_totals = {
        key: int(sum(int(item.get(key, 0)) for item in hsp_items))
        for key in action_keys
    }
    event_errors = [event for event in events if event.get("error")]
    truncated_feedback = [event for event in events if event.get("feedback_truncated")]
    collection_mode_counts = Counter(str(item.get("collection_mode", "policy")) for item in hsp_items)
    data_role_counts = Counter(str(item.get("data_role", "unspecified")) for item in hsp_items)

    summary: dict[str, Any] = {
        "source": source,
        "examples": len(hsp_items),
        "mean_score": _mean(float(item.get("score", 0.0)) for item in hsp_items),
        "teacher_tokens_total": int(sum(teacher_tokens)),
        "teacher_tokens_mean": _mean(teacher_tokens),
        "action_totals": action_totals,
        "actions_per_example": {
            key: action_totals[key] / len(hsp_items) for key in action_keys
        },
        "event_error_count": len(event_errors),
        "feedback_truncated_count": len(truncated_feedback),
        "collection_mode_counts": dict(sorted(collection_mode_counts.items())),
        "data_role_counts": dict(sorted(data_role_counts.items())),
        "groups": {
            "no_interaction": _subset_metrics(
                hsp_items,
                lambda item: int(item.get("ask_count", 0)) + int(item.get("verify_count", 0)) == 0,
            ),
            "any_interaction": _subset_metrics(
                hsp_items,
                lambda item: int(item.get("ask_count", 0)) + int(item.get("verify_count", 0)) > 0,
            ),
            "asked": _subset_metrics(hsp_items, lambda item: int(item.get("ask_count", 0)) > 0),
            "verified": _subset_metrics(hsp_items, lambda item: int(item.get("verify_count", 0)) > 0),
            "accepted": _subset_metrics(hsp_items, lambda item: int(item.get("accept_count", 0)) > 0),
            "verified_without_accept": _subset_metrics(
                hsp_items,
                lambda item: int(item.get("verify_count", 0)) > 0 and int(item.get("accept_count", 0)) == 0,
            ),
        },
    }
    return summary


def _utility(item: dict[str, Any], teacher_token_budget: float, teacher_cost_weight: float) -> float:
    return float(item.get("score", 0.0)) - teacher_cost_weight * (
        float(item.get("teacher_tokens_used", 0)) / teacher_token_budget
    )


def summarize_paired_calibration(
    items: list[dict[str, Any]],
    teacher_token_budget: float = 192.0,
    teacher_cost_weight: float = 0.15,
) -> dict[str, Any]:
    if teacher_token_budget <= 0:
        raise ValueError("teacher_token_budget must be positive.")
    hsp_items = [item for item in items if item.get("interaction_policy") == "hsp"]
    paired: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in hsp_items:
        key = (str(item.get("question") or item.get("source_id") or item.get("id") or ""), int(item.get("sample_index", 0)))
        if key[0]:
            paired[key][str(item.get("collection_mode", "policy"))].append(item)

    comparisons: dict[str, dict[str, Any]] = {}
    modes = sorted({str(item.get("collection_mode", "policy")) for item in hsp_items}.difference({"independent"}))
    for mode in modes:
        score_deltas: list[float] = []
        utility_deltas: list[float] = []
        rescued = 0
        unnecessary_interactions = 0
        for runs in paired.values():
            if "independent" not in runs or mode not in runs:
                continue
            baseline_score = _mean(float(item.get("score", 0.0)) for item in runs["independent"])
            assisted_score = _mean(float(item.get("score", 0.0)) for item in runs[mode])
            baseline_utility = _mean(_utility(item, teacher_token_budget, teacher_cost_weight) for item in runs["independent"])
            assisted_utility = _mean(_utility(item, teacher_token_budget, teacher_cost_weight) for item in runs[mode])
            assert baseline_score is not None and assisted_score is not None
            assert baseline_utility is not None and assisted_utility is not None
            score_deltas.append(assisted_score - baseline_score)
            utility_deltas.append(assisted_utility - baseline_utility)
            if baseline_score <= 0.0 and assisted_score > 0.0:
                rescued += 1
            if baseline_score > 0.0 and any(
                int(item.get("ask_count", 0)) + int(item.get("verify_count", 0)) > 0 for item in runs[mode]
            ):
                unnecessary_interactions += 1
        comparisons[mode] = {
            "paired_examples": len(score_deltas),
            "mean_score_delta_vs_independent": _mean(score_deltas),
            "mean_utility_delta_vs_independent": _mean(utility_deltas),
            "rescued_failures": rescued,
            "interactions_on_independent_successes": unnecessary_interactions,
        }

    verify_events = [
        event
        for item in hsp_items
        for event in item.get("events", [])
        if event.get("action") == "verify"
    ]
    return {
        "comparisons_vs_independent": comparisons,
        "verify_trust": {
            "review_events": len(verify_events),
            "explicit_accepts": sum(bool(event.get("accepted", False)) for event in verify_events),
            "implicit_adoptions_without_accept": sum(
                bool(event.get("implicit_adoption_without_accept", False)) for event in verify_events
            ),
        },
    }


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of per-example evaluation results.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one or more results_*_hsp.json files.")
    parser.add_argument("inputs", nargs="+", help="HSP per-example JSON result files.")
    parser.add_argument("--output", help="Optional JSON path for the aggregated summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = [(path, load_results(Path(path))) for path in args.inputs]
    summaries = [summarize_items(items, source=path) for path, items in loaded]
    report = {
        "runs": summaries,
        "paired_calibration": summarize_paired_calibration([item for _, items in loaded for item in items]),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
