"""Combine protocol cold-start and outcome-selected replay HSP data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

try:
    from .preflight_hsp import load_examples, validate_dataset
except ImportError:
    from preflight_hsp import load_examples, validate_dataset


def mix_examples(
    protocol_examples: list[dict[str, Any]],
    replay_examples: list[dict[str, Any]],
    max_replay_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    if not protocol_examples:
        raise ValueError("Protocol data must not be empty.")
    if not 0 <= max_replay_fraction < 1:
        raise ValueError("max_replay_fraction must be in [0, 1).")

    if max_replay_fraction == 0:
        selected_replay: list[dict[str, Any]] = []
    else:
        replay_limit = int(len(protocol_examples) * max_replay_fraction / (1 - max_replay_fraction))
        if len(replay_examples) <= replay_limit:
            selected_replay = list(replay_examples)
        else:
            selected_replay = rng.sample(replay_examples, replay_limit)

    mixed = list(protocol_examples) + selected_replay
    rng.shuffle(mixed)
    return mixed, len(selected_replay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix HSP cold-start and replay SFT datasets.")
    parser.add_argument("--protocol_data", required=True, help="Synthetic protocol seed JSON/JSONL.")
    parser.add_argument("--replay_data", required=True, help="Outcome-selected replay JSON/JSONL.")
    parser.add_argument("--output", required=True, help="Mixed HSP training JSONL.")
    parser.add_argument(
        "--max_replay_fraction",
        type=float,
        default=0.50,
        help="Maximum fraction of replay examples after mixing; protocol examples are all retained.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_examples = load_examples(args.protocol_data)
    replay_examples = load_examples(args.replay_data)
    mixed, selected_replay_count = mix_examples(
        protocol_examples,
        replay_examples,
        args.max_replay_fraction,
        random.Random(args.seed),
    )
    report = validate_dataset(mixed)
    if report["errors"]:
        raise ValueError("Mixed HSP data failed validation: " + "; ".join(report["errors"]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for example in mixed:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")

    fraction = selected_replay_count / len(mixed) if mixed else 0.0
    print(f"Wrote {len(mixed)} examples to {output_path}.")
    print(f"Protocol examples retained: {len(protocol_examples)}.")
    print(f"Replay examples selected: {selected_replay_count} ({fraction:.4f} of mixed data).")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
