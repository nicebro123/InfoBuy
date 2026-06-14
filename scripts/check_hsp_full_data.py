#!/usr/bin/env python3
"""Validate that InfoBuy full-data experiments will not silently use smoke data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


REQUIRED_JSONL = {
    "raw/numinamath_cot_synthetic_math_seed_v0_1500.jsonl": 1500,
    "raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl": 800,
    "raw/numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl": 200,
    "protocol/hsp_protocol_train_pilot_v1.jsonl": 4800,
    "protocol/hsp_protocol_validation_pilot_v1.jsonl": 1200,
}

OPTIONAL_JSONL = {
    "protocol/hsp_protocol_train_pilot_v1_action_boost_x3.jsonl": 12000,
}

EXPECTED_EVAL_COUNTS = {
    "math": 500,
    "gsm8k": 1319,
    "minerva": 272,
    "olympiad": 675,
    "aime2024": 30,
    "aime2025": 30,
}


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def validate_jsonl(root: Path, entries: dict[str, int], *, required: bool) -> list[str]:
    errors: list[str] = []
    for rel_path, expected in entries.items():
        path = root / rel_path
        if not path.exists():
            message = f"MISSING {path}"
            if required:
                errors.append(message)
            print(message)
            continue
        actual = count_jsonl(path)
        status = "OK" if actual == expected else "BAD"
        print(f"{status} {path} rows={actual} expected={expected}")
        if actual != expected:
            errors.append(f"{path} rows={actual}, expected={expected}")
    return errors


def validate_record_schema(root: Path) -> list[str]:
    errors: list[str] = []
    checks = [
        root / "raw" / "numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl",
        root / "raw" / "numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl",
    ]
    for path in checks:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as stream:
            first = json.loads(next(line for line in stream if line.strip()))
        missing = [key for key in ("question", "gold_answer") if key not in first]
        if missing:
            errors.append(f"{path} missing keys: {', '.join(missing)}")
        else:
            print(f"OK schema {path} has question/gold_answer")
    return errors


def validate_eval_sets() -> list[str]:
    errors: list[str] = []
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from eval import datasets_loader
    except Exception as exc:
        return [f"failed to import eval.datasets_loader: {exc!r}"]
    for name, expected in EXPECTED_EVAL_COUNTS.items():
        try:
            handler = datasets_loader.get_dataset_handler(name)
            questions, answers = handler.load_data()
        except Exception as exc:
            errors.append(f"{name} failed to load: {exc!r}")
            continue
        actual = len(questions)
        status = "OK" if actual == expected and len(answers) == expected else "BAD"
        print(f"{status} eval {name} rows={actual} expected={expected}")
        if actual != expected or len(answers) != expected:
            errors.append(f"{name} rows={actual}/{len(answers)}, expected={expected}")
    return errors


def validate_math500_local(data_root: Path) -> list[str]:
    path = data_root / "benchmarks" / "math500" / "math_500_test.csv"
    if not path.exists():
        return [f"MISSING local MATH500 CSV: {path}"]
    print(f"OK local MATH500 CSV {path}")
    return []


def print_header(title: str) -> None:
    print()
    print(f"== {title} ==")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Fail on optional file mismatches too.")
    args = parser.parse_args(argv)

    generated = os.environ.get("INFOBUY_GENERATED_DATA")
    datasets_root = os.environ.get("INFOBUY_DATASETS")
    if not generated:
        raise SystemExit("INFOBUY_GENERATED_DATA is not set; source setup/env.sh first.")
    if not datasets_root:
        raise SystemExit("INFOBUY_DATASETS is not set; source setup/env.sh first.")

    generated_root = Path(generated)
    data_root = Path(datasets_root)
    errors: list[str] = []

    print_header("Required full train data")
    errors.extend(validate_jsonl(generated_root, REQUIRED_JSONL, required=True))
    errors.extend(validate_record_schema(generated_root))

    print_header("Optional protocol variants")
    optional_errors = validate_jsonl(generated_root, OPTIONAL_JSONL, required=False)
    if args.strict:
        errors.extend(optional_errors)

    print_header("Full evaluation data")
    errors.extend(validate_math500_local(data_root))
    errors.extend(validate_eval_sets())

    if errors:
        print_header("FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print_header("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
