"""Calibrate HSP source-problem difficulty with a strong reference model.

Purpose differs by training stage
----------------------------------
The HSP **cold-start SFT** dataset exists to teach the model the *protocol* -- when
and how to emit ``<ASK>N</ASK>`` / ``<VERIFY>N</VERIFY>`` / ``<ACCEPT>``. For that you
want BOTH easy and hard problems, so the tokens are learned across the full difficulty
range. **Do not narrow SFT data to a band** -- keep the spread (``--mode tag``, the
default; or ``--mode stratify`` for balanced coverage).

The narrow ~70% difficulty band matters later, for the **RL stage**, where the actual
help-seeking *decisions* are optimized and problems outside the band carry little signal
(``--mode target --target_accuracy 0.70``).

What it does
------------
1. Runs a reference model (default: Qwen/Qwen3-8B) over each problem, ``k`` samples.
2. Scores every sample against ``gold_answer`` with math-equivalence checking.
3. Tags each problem with ``n_samples`` / ``n_correct`` / ``solve_rate``.
4. Reports the aggregate accuracy (= mean solve_rate) and the difficulty histogram.
5. Selects a subset according to ``--mode``:
   * ``tag``      -- keep ALL problems, just annotate difficulty (default; for SFT).
   * ``stratify`` -- balance easy/medium/hard coverage (for SFT, guarantees spread).
   * ``target``   -- greedily approach ``--target_accuracy`` (for RL-stage data).
   * ``band``     -- keep an explicit ``--min_solve_rate``/``--max_solve_rate`` band.
6. Writes the (tagged/selected) JSONL plus a JSON report.

Typical use (on a GPU box)
--------------------------
    # SFT cold-start: keep easy AND hard, just tag difficulty (recommended)
    python -m SFT_stage.calibrate_hsp_difficulty \
        --input  "$INFOBUY_GENERATED_DATA/raw/numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl" \
        --output "$INFOBUY_GENERATED_DATA/raw/numinamath_cot_synthetic_math_train_calibrated_800.jsonl" \
        --report "$INFOBUY_GENERATED_DATA/manifests/difficulty_train.json" \
        --model_path Qwen/Qwen3-8B --samples_k 4 --mode tag

    # RL stage: narrow to the ~70% help-seeking band
    python -m SFT_stage.calibrate_hsp_difficulty ... --mode target --target_accuracy 0.70

This module imports vLLM lazily, so ``--help`` and the pure-python selection
logic remain importable (and unit-testable) on a machine without a GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def user_prompt(question: str) -> str:
    if "Please reason step by step" in question:
        return question
    return question.rstrip() + "\nPlease reason step by step, and put your final answer within \\boxed{}."


# --------------------------------------------------------------------------
# Answer scoring (math-equivalence with a safe fallback)
# --------------------------------------------------------------------------

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


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).strip().rstrip(".").lower()


def make_scorer():
    """Return an ``is_correct(response, gold) -> bool`` callable.

    Prefers ``math_verify`` for symbolic equivalence; falls back to a normalized
    string comparison of the last boxed answer if the library is unavailable.
    """
    try:
        from math_verify import parse, verify

        def is_correct(response: str, gold: str) -> bool:
            predicted = extract_last_boxed(response)
            if predicted is None:
                return False
            try:
                return bool(verify(parse(str(gold)), parse(str(predicted))))
            except Exception:
                return _normalize(predicted) == _normalize(gold)

        return is_correct, "math_verify"
    except ImportError:
        def is_correct(response: str, gold: str) -> bool:
            predicted = extract_last_boxed(response)
            if predicted is None:
                return False
            return _normalize(predicted) == _normalize(gold)

        return is_correct, "string_fallback"


# --------------------------------------------------------------------------
# Difficulty selection (pure python; unit-testable without a GPU)
# --------------------------------------------------------------------------

def aggregate_accuracy(records: list[dict[str, Any]]) -> float:
    rates = [r["solve_rate"] for r in records if "solve_rate" in r]
    return sum(rates) / len(rates) if rates else 0.0


def difficulty_bin(rate: float) -> str:
    if rate == 0.0:
        return "0.0"
    if rate <= 0.25:
        return "(0,0.25]"
    if rate <= 0.5:
        return "(0.25,0.5]"
    if rate <= 0.75:
        return "(0.5,0.75]"
    if rate < 1.0:
        return "(0.75,1.0)"
    return "1.0"


def difficulty_histogram(records: list[dict[str, Any]]) -> dict[str, int]:
    bins = {"0.0": 0, "(0,0.25]": 0, "(0.25,0.5]": 0, "(0.5,0.75]": 0, "(0.75,1.0)": 0, "1.0": 0}
    for record in records:
        rate = record.get("solve_rate")
        if rate is None:
            continue
        bins[difficulty_bin(rate)] += 1
    return bins


def filter_band(
    records: list[dict[str, Any]],
    min_solve_rate: float,
    max_solve_rate: float,
) -> list[dict[str, Any]]:
    """Keep problems whose solve_rate lies in the inclusive [min, max] band."""
    return [r for r in records if min_solve_rate <= r.get("solve_rate", -1.0) <= max_solve_rate]


def stratify_balanced(
    records: list[dict[str, Any]],
    max_per_bin: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Keep a balanced mix across difficulty bins so easy AND hard are both present.

    Groups problems into the same difficulty bins as the histogram, then keeps up to
    ``max_per_bin`` from each (defaults to the size of the smallest non-empty bin, i.e.
    a fully balanced spread). This preserves both easy and hard problems -- the right
    behavior for protocol/token-learning SFT data.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        rate = record.get("solve_rate")
        if rate is None:
            continue
        grouped[difficulty_bin(rate)].append(record)
    if not grouped:
        return []
    if max_per_bin is None:
        max_per_bin = min(len(items) for items in grouped.values())
    rng = random.Random(seed)
    kept: list[dict[str, Any]] = []
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        kept.extend(items[:max_per_bin])
    return kept


def select_to_target(
    records: list[dict[str, Any]],
    target_accuracy: float,
    tolerance: float = 0.02,
    drop_unsolved: bool = True,
) -> list[dict[str, Any]]:
    """Greedily compose the largest subset whose mean solve_rate ~= target.

    Strategy: optionally drop solve_rate==0 problems first (these usually flag a
    wrong gold answer or out-of-band difficulty and add no teachable help signal),
    then iteratively remove the single most-extreme problem that moves the mean
    toward the target until we land within ``tolerance``.
    """
    kept = list(records)
    if drop_unsolved:
        kept = [r for r in kept if r.get("solve_rate", 0.0) > 0.0]
    if not kept:
        return kept

    kept.sort(key=lambda r: r["solve_rate"])  # hardest first
    while len(kept) > 1:
        mean = aggregate_accuracy(kept)
        if abs(mean - target_accuracy) <= tolerance:
            break
        if mean > target_accuracy:
            # too easy: drop the easiest (last) problem to add difficulty
            kept.pop()
        else:
            # too hard: drop the hardest (first) problem to raise accuracy
            kept.pop(0)
    return kept


# --------------------------------------------------------------------------
# Model rollout (vLLM, imported lazily)
# --------------------------------------------------------------------------

def score_problems(
    records: list[dict[str, Any]],
    model_path: str,
    samples_k: int,
    temperature: float,
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> tuple[list[dict[str, Any]], str]:
    import vllm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = vllm.LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    is_correct, scorer_name = make_scorer()

    prompts: list[str] = []
    for record in records:
        question = str(record.get("question", ""))
        messages = [{"role": "user", "content": user_prompt(question)}]
        if getattr(tokenizer, "chat_template", None):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            rendered = "User: " + user_prompt(question) + "\nAssistant:"
        prompts.append(rendered)

    sampling_params = vllm.SamplingParams(
        n=samples_k,
        temperature=temperature if samples_k > 1 else 0.0,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)

    for record, output in zip(records, outputs):
        gold = str(record.get("gold_answer", ""))
        completions = [c.text for c in output.outputs]
        n_correct = sum(1 for text in completions if is_correct(text, gold))
        record["n_samples"] = len(completions)
        record["n_correct"] = n_correct
        record["solve_rate"] = (n_correct / len(completions)) if completions else 0.0
        record["calibration_model"] = model_path
    return records, scorer_name


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Raw problems JSONL (needs question + gold_answer).")
    parser.add_argument("--output", required=True, help="Calibrated (filtered) JSONL output path.")
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    parser.add_argument("--tagged_output", default=None,
                        help="Optional path to write ALL problems with difficulty tags (pre-filter).")
    parser.add_argument("--model_path", default="Qwen/Qwen3-8B", help="Reference model for difficulty scoring.")
    parser.add_argument("--samples_k", type=int, default=4, help="Samples per problem.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_records", type=int, default=None, help="Limit problems (debugging).")

    # Selection mode
    parser.add_argument("--mode", choices=["tag", "stratify", "target", "band"], default="tag",
                        help="tag=keep all (SFT default); stratify=balance easy/hard (SFT); "
                             "target=approach --target_accuracy (RL); band=explicit min/max.")
    parser.add_argument("--max_per_bin", type=int, default=None,
                        help="[stratify] Max problems kept per difficulty bin (default: smallest bin = balanced).")
    parser.add_argument("--target_accuracy", type=float, default=0.70,
                        help="[target] Compose the largest subset whose mean solve_rate ~= this (e.g. 0.70).")
    parser.add_argument("--tolerance", type=float, default=0.02, help="[target] Tolerance for --target_accuracy.")
    parser.add_argument("--min_solve_rate", type=float, default=0.0,
                        help="[band] Keep solve_rate >= this.")
    parser.add_argument("--max_solve_rate", type=float, default=1.0,
                        help="[band] Keep solve_rate <= this.")
    parser.add_argument("--keep_unsolved", action="store_true",
                        help="[target] Keep solve_rate==0 problems (default drops them as likely broken/too-hard).")
    parser.add_argument("--seed", type=int, default=0, help="[stratify] Shuffle seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.input))
    if args.max_records is not None:
        records = records[: args.max_records]
    if not records:
        raise ValueError(f"No records loaded from {args.input}.")

    records, scorer_name = score_problems(
        records,
        model_path=args.model_path,
        samples_k=args.samples_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    before_accuracy = aggregate_accuracy(records)
    before_hist = difficulty_histogram(records)

    if args.tagged_output:
        write_records(Path(args.tagged_output), records)

    if args.mode == "tag":
        kept = records
        mode = "tag (keep all; SFT default)"
    elif args.mode == "stratify":
        kept = stratify_balanced(records, max_per_bin=args.max_per_bin, seed=args.seed)
        mode = f"stratify (max_per_bin={args.max_per_bin or 'balanced'})"
    elif args.mode == "target":
        kept = select_to_target(
            records, args.target_accuracy, tolerance=args.tolerance,
            drop_unsolved=not args.keep_unsolved,
        )
        mode = f"target_accuracy={args.target_accuracy}"
    else:  # band
        kept = filter_band(records, args.min_solve_rate, args.max_solve_rate)
        mode = f"band=[{args.min_solve_rate},{args.max_solve_rate}]"

    write_records(Path(args.output), kept)
    after_accuracy = aggregate_accuracy(kept)
    after_hist = difficulty_histogram(kept)

    report = {
        "input": args.input,
        "output": args.output,
        "model_path": args.model_path,
        "scorer": scorer_name,
        "samples_k": args.samples_k,
        "selection_mode": mode,
        "n_total": len(records),
        "n_kept": len(kept),
        "accuracy_before": round(before_accuracy, 4),
        "accuracy_after_kept": round(after_accuracy, 4),
        "histogram_before": before_hist,
        "histogram_after": after_hist,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"\nReference model ({args.model_path}, scorer={scorer_name}) solved "
        f"{before_accuracy*100:.1f}% of {len(records)} problems."
    )
    print(
        f"Kept {len(kept)}/{len(records)} problems; kept-set accuracy {after_accuracy*100:.1f}% "
        f"(mode: {mode})."
    )
    if args.mode in {"tag", "stratify"}:
        easy = after_hist["1.0"]
        hard = after_hist["0.0"] + after_hist["(0,0.25]"]
        print(f"Difficulty spread kept: {easy} easy (solve_rate=1.0), {hard} hard (solve_rate<=0.25). "
              f"Both easy and hard are retained for protocol/token learning.")


if __name__ == "__main__":
    main()
