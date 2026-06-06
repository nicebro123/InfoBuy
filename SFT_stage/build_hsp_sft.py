"""Build cold-start Help-Seeking Policy (HSP) SFT trajectories.

This builder deliberately creates protocol supervision, not calibrated
help-seeking decisions. Later stages should replace or augment these examples
with student rollouts and teacher-validated decisions.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


SAMPLE_WEIGHTS = {
    "normal": 0.45,
    "ask_help": 0.15,
    "verify_confirm": 0.10,
    "verify_accept_correction": 0.15,
    "verify_reject_bad_feedback": 0.10,
    "verify_uncertain": 0.05,
}

# Token budget ranges for <ASK>N</ASK> and <VERIFY>N</VERIFY>
# The student model learns to request different amounts of help
ASK_BUDGET_CHOICES = [32, 64, 96, 128]
VERIFY_BUDGET_CHOICES = [64, 96, 128]
PROVENANCE_FIELDS = (
    "source_dataset",
    "source_config",
    "source_split",
    "source_category",
    "source_row_index",
    "source_license",
    "source_url",
)
# Strip any reserved control marker from source prompts/solutions. This handles
# both the budgeted form (<ASK>N</ASK>, <VERIFY>N</VERIFY>, <call>N</call>) and
# any stray bare tags so that source text never leaks protocol tokens.
CONTROL_PATTERN = re.compile(
    r"<call>\s*\d*\s*</call>"
    r"|<ASK>\s*\d*\s*</ASK>"
    r"|<VERIFY>\s*\d*\s*</VERIFY>"
    r"|</?(?:ASK|VERIFY|ACCEPT|TEACHER_HELP|TEACHER_REVIEW|ENVIRONMENT_NOTICE)>",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create structured HSP SFT JSONL data.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Input JSON or JSONL with questions and solutions.")
    source.add_argument("--dataset_name", help="Hugging Face dataset ID with questions and solutions.")
    parser.add_argument("--dataset_config", default=None, help="Optional Hugging Face dataset configuration.")
    parser.add_argument("--dataset_split", default="train", help="Hugging Face dataset split.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument(
        "--variants_per_problem",
        type=int,
        default=1,
        help="Weighted sample variants generated per problem unless --emit_all_types is set.",
    )
    parser.add_argument(
        "--emit_all_types",
        action="store_true",
        help="Emit all six protocol sample types for every usable problem.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    with path.open("r", encoding="utf-8") as source:
        data = json.load(source)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("train", "data", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Input must be a list or contain a list under train/data/examples.")


def load_hub_records(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    max_records: int | None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("Reading a Hugging Face dataset requires the datasets package.") from error

    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    if max_records is not None:
        dataset = dataset.select(range(min(max_records, len(dataset))))
    return [dict(record) for record in dataset]


def extract_last_boxed(text: str) -> str | None:
    last_value = None
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = text.find("{", match.start())
        depth = 0
        for position in range(start, len(text)):
            char = text[position]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    last_value = text[start + 1 : position].strip()
                    break
    return last_value


def clean_solution(text: str) -> str:
    return CONTROL_PATTERN.sub("", text).strip()


def clean_gold_answer(text: str) -> str:
    return text.strip().rstrip(".,;:").strip()


def get_first(record: dict[str, Any], names: Iterable[str]) -> str | None:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def normalize_record(record: dict[str, Any], index: int) -> dict[str, Any] | None:
    question = get_first(record, ("question", "problem", "prompt"))
    raw_solution = get_first(record, ("gold_solution", "solution", "response", "answer"))
    if not question or not raw_solution:
        return None

    question = clean_solution(question)
    solution = clean_solution(raw_solution)
    if not question:
        return None
    gold_answer = get_first(record, ("gold_answer", "final_answer", "target"))
    if not gold_answer:
        gold_answer = extract_last_boxed(solution)
    if not gold_answer and "####" in solution:
        gold_answer = solution.rsplit("####", 1)[-1].strip()
    if not gold_answer and len(solution) <= 80:
        gold_answer = solution.strip()
    if not gold_answer:
        return None
    gold_answer = clean_gold_answer(gold_answer)
    if not gold_answer:
        return None

    if extract_last_boxed(solution) != gold_answer:
        solution = f"{solution}\nTherefore, the final answer is \\boxed{{{gold_answer}}}."
    normalized: dict[str, Any] = {
        "id": str(record.get("id", f"record_{index:07d}")),
        "question": question,
        "gold_answer": gold_answer,
        "gold_solution": solution,
    }
    for field in PROVENANCE_FIELDS:
        if field in record and record[field] is not None:
            normalized[field] = record[field]
    return normalized


def user_prompt(question: str) -> str:
    if "Please reason step by step" in question:
        return question
    return question.rstrip() + "\nPlease reason step by step, and put your final answer within \\boxed{}."


def student_segment(text: str) -> dict[str, Any]:
    return {"source": "student", "text": text, "loss": True}


def context_segment(source: str, text: str) -> dict[str, Any]:
    return {"source": source, "text": text, "loss": False}


def derive_help_hint(solution: str) -> str:
    lines = [line.strip() for line in solution.splitlines() if line.strip() and "\\boxed" not in line]
    hint = lines[0] if lines else "Set up the governing equation before simplifying."
    return hint[:240]


def incorrect_answer(gold_answer: str) -> str:
    compact = gold_answer.replace(",", "").strip()
    if re.fullmatch(r"[-+]?\d+", compact):
        return str(int(compact) + 1)
    if re.fullmatch(r"[-+]?\d*\.\d+", compact):
        return str(float(compact) + 1.0)
    return f"({gold_answer})+1"


def make_example(
    problem: dict[str, Any], sample_type: str, variant: int, rng: random.Random | None = None,
) -> dict[str, Any]:
    question = problem["question"]
    gold = problem["gold_answer"]
    solution = problem["gold_solution"]
    wrong = incorrect_answer(gold)
    segments = [context_segment("user", user_prompt(question))]
    feedback_is_correct: bool | None = None

    # Pick token budgets for <ASK>N</ASK> and <VERIFY>N</VERIFY>
    _rng = rng or random.Random()
    ask_n = _rng.choice(ASK_BUDGET_CHOICES)
    verify_n = _rng.choice(VERIFY_BUDGET_CHOICES)

    if sample_type == "normal":
        segments.append(student_segment(solution))
    elif sample_type == "ask_help":
        segments.extend(
            [
                student_segment(
                    f"I need a useful next reasoning step before completing this solution.\n<ASK>{ask_n}</ASK>"
                ),
                context_segment(
                    "teacher",
                    f"\n<TEACHER_HELP>\nA useful next step is: {derive_help_hint(solution)}\n</TEACHER_HELP>\n",
                ),
                student_segment(solution),
            ]
        )
    elif sample_type == "verify_confirm":
        feedback_is_correct = True
        segments.extend(
            [
                student_segment(f"{solution}\n<VERIFY>{verify_n}</VERIFY>"),
                context_segment(
                    "teacher",
                    "\n<TEACHER_REVIEW>\nVerdict: correct\nIssue: None\nCorrection: None\n"
                    f"Suggested answer: \\boxed{{{gold}}}\n</TEACHER_REVIEW>\n",
                ),
                student_segment(f"The checked final answer remains \\boxed{{{gold}}}."),
            ]
        )
    elif sample_type == "verify_accept_correction":
        feedback_is_correct = True
        segments.extend(
            [
                student_segment(f"A tentative calculation gives \\boxed{{{wrong}}}.\n<VERIFY>{verify_n}</VERIFY>"),
                context_segment(
                    "teacher",
                    "\n<TEACHER_REVIEW>\nVerdict: incorrect\n"
                    f"Issue: The tentative result {wrong} does not match the corrected derivation.\n"
                    f"Correction: The corrected result is \\boxed{{{gold}}}.\n"
                    f"Suggested answer: \\boxed{{{gold}}}\n</TEACHER_REVIEW>\n",
                ),
                student_segment(f"<ACCEPT>\n{solution}"),
            ]
        )
    elif sample_type == "verify_reject_bad_feedback":
        feedback_is_correct = False
        segments.extend(
            [
                student_segment(f"{solution}\n<VERIFY>{verify_n}</VERIFY>"),
                context_segment(
                    "teacher",
                    "\n<TEACHER_REVIEW>\nVerdict: incorrect\n"
                    f"Issue: Replace the final value with {wrong}.\n"
                    f"Correction: The final value is \\boxed{{{wrong}}}.\n"
                    f"Suggested answer: \\boxed{{{wrong}}}\n</TEACHER_REVIEW>\n",
                ),
                student_segment(
                    f"The proposed correction conflicts with the derivation, so I retain \\boxed{{{gold}}}."
                ),
            ]
        )
    elif sample_type == "verify_uncertain":
        feedback_is_correct = None
        segments.extend(
            [
                student_segment(
                    f"I have a tentative derivation but want an external check before finalizing.\n<VERIFY>{verify_n}</VERIFY>"
                ),
                context_segment(
                    "teacher",
                    "\n<TEACHER_REVIEW>\nVerdict: uncertain\nIssue: The available check is inconclusive.\n"
                    "Correction: None\nSuggested answer: None\n</TEACHER_REVIEW>\n",
                ),
                student_segment(solution),
            ]
        )
    else:
        raise ValueError(f"Unsupported sample type: {sample_type}")

    example = {
        "id": f"{problem['id']}_{sample_type}_{variant}",
        "source_id": problem["id"],
        "sample_type": sample_type,
        "generation_mode": "synthetic_protocol_seed",
        "gold_answer": gold,
        "feedback_is_correct": feedback_is_correct,
        "segments": segments,
    }
    for field in PROVENANCE_FIELDS:
        if field in problem:
            example[field] = problem[field]
    return example


def build_examples(
    records: list[dict[str, Any]],
    rng: random.Random,
    variants_per_problem: int,
    emit_all_types: bool,
) -> tuple[list[dict[str, Any]], int]:
    examples: list[dict[str, Any]] = []
    skipped = 0
    types = list(SAMPLE_WEIGHTS)
    weights = list(SAMPLE_WEIGHTS.values())
    for index, record in enumerate(records):
        problem = normalize_record(record, index)
        if problem is None:
            skipped += 1
            continue
        chosen_types = types if emit_all_types else rng.choices(types, weights=weights, k=variants_per_problem)
        for variant, sample_type in enumerate(chosen_types):
            examples.append(make_example(problem, sample_type, variant, rng=rng))
    return examples, skipped


def main() -> None:
    args = parse_args()
    if args.variants_per_problem <= 0:
        raise ValueError("--variants_per_problem must be greater than zero.")
    if args.dataset_name:
        records = load_hub_records(args.dataset_name, args.dataset_config, args.dataset_split, args.max_records)
    else:
        records = load_records(Path(args.input))
    if args.max_records is not None and not args.dataset_name:
        records = records[: args.max_records]
    examples, skipped = build_examples(
        records,
        random.Random(args.seed),
        variants_per_problem=args.variants_per_problem,
        emit_all_types=args.emit_all_types,
    )
    if not examples:
        raise ValueError("No usable examples were generated. Check that records include questions and gold solutions.")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for example in examples:
            output.write(json.dumps(example, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for example in examples:
        counts[example["sample_type"]] = counts.get(example["sample_type"], 0) + 1
    print(f"Wrote {len(examples)} examples to {output_path}. Skipped {skipped} unusable records.")
    print("Sample counts: " + json.dumps(counts, sort_keys=True))
    print("Note: generation_mode=synthetic_protocol_seed teaches protocol usage, not calibrated calling decisions.")


if __name__ == "__main__":
    main()
