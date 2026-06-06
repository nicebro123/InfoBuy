"""Optionally recheck incorrect evaluation answers with an external judge model."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
from pathlib import Path

from tqdm import tqdm


MAX_WORKERS = 10
DATASETS_TO_CHECK = ["math", "gsm8k", "amc", "minerva", "olympiad", "aime2024", "aime2025"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--larger_model", type=str, default="-")
    parser.add_argument("--fix_number", type=int, default=None)
    parser.add_argument("--interaction_policy", choices=["relay_call", "hsp"], default="relay_call")
    parser.add_argument(
        "--collection_mode",
        choices=["policy", "independent", "force_ask_first", "force_verify_after_draft"],
        default="policy",
    )
    parser.add_argument("--output_tag", default=None, help="Optional generation run tag used in result filenames.")
    parser.add_argument(
        "--skip_llm_recheck",
        action="store_true",
        help="Keep deterministic grader scores and do not call an external answer judge.",
    )
    parser.add_argument("--judge_model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--overwrite_recheck",
        action="store_true",
        help="Explicitly replace existing rechecked result sidecars.",
    )
    return parser.parse_args()


def safe_output_component(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return rendered[:64] or "run"


def policy_suffix(args: argparse.Namespace) -> str:
    if args.interaction_policy == "relay_call":
        return ""
    if args.collection_mode == "policy":
        return "_hsp"
    return f"_hsp_{args.collection_mode}"


def result_filename(dataset: str, args: argparse.Namespace) -> str:
    dataset_key = dataset
    if args.output_tag:
        dataset_key += "_" + safe_output_component(args.output_tag)
    fix_suffix = f"_{args.fix_number}" if args.fix_number is not None else ""
    return f"results_{dataset_key}{policy_suffix(args)}{fix_suffix}.json"


def create_judge_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for LLM answer recheck. "
            "Set it or pass --skip_llm_recheck to report deterministic grader scores only."
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the openai package or pass --skip_llm_recheck.") from error
    kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def process_example(client, judge_model: str, answer: str, response: str) -> str:
    completion = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": "You are a math answer checker. Reply only Yes or No."},
            {
                "role": "user",
                "content": (
                    f"Reference answer:\n{answer}\n\n"
                    f"Candidate response:\n{response}\n\n"
                    "Is the candidate answer mathematically equivalent to the reference answer?"
                ),
            },
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content.strip()


def normalize_judgement(text: str) -> str:
    match = re.match(r"^\s*(yes|no)\b", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unexpected judge response: {text!r}")
    return match.group(1).lower()


def load_results(storage_path: str, model_name: str, larger_model: str, filename: str):
    primary_path = Path(storage_path) / "evaluation" / f"{model_name.replace('/', '_')}_{larger_model}" / filename
    fallback_path = Path(storage_path) / "evaluation" / model_name.replace("/", "_") / filename
    for candidate in (primary_path, fallback_path):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as stream:
                return json.load(stream), candidate
    return None, primary_path


def rechecked_results_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}_rechecked{source_path.suffix}")


def rechecked_summary_path(per_item_path: Path) -> Path:
    return per_item_path.with_name(f"{per_item_path.stem}_summary.json")


def ensure_recheck_outputs_available(source_path: Path, overwrite: bool) -> None:
    per_item_path = rechecked_results_path(source_path)
    output_paths = (per_item_path, rechecked_summary_path(per_item_path))
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Rechecked output already exists: "
            + ", ".join(existing)
            + ". Pass --overwrite_recheck explicitly or use a new --output_tag."
        )


def write_rechecked_results(source_path: Path, results: list[dict], overwrite: bool = False) -> Path:
    output_path = rechecked_results_path(source_path)
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    return output_path


def write_rechecked_summary(per_item_path: Path, result_entry: dict, overwrite: bool = False) -> Path:
    output_path = rechecked_summary_path(per_item_path)
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8") as stream:
        json.dump(result_entry, stream, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    args = parse_args()
    storage_path = os.getenv("STORAGE_PATH", ".")
    judge_client = None
    new_results = []

    for dataset in DATASETS_TO_CHECK:
        print(f"\n--- Processing {args.model_name} on {dataset} ---")
        results, source_path = load_results(
            storage_path, args.model_name, args.larger_model, result_filename(dataset, args)
        )
        if results is None:
            print(f"Result file not found: {source_path}")
            continue
        if not results:
            print(f"No results loaded for {dataset}, skipping.")
            continue
        overwrite_recheck = bool(getattr(args, "overwrite_recheck", False))
        ensure_recheck_outputs_available(source_path, overwrite_recheck)
        print(f"Loaded results from {source_path}")
        for result in results:
            result["deterministic_score"] = result.get("score", 0)
            result["llm_recheck_judgement"] = None

        items_to_recheck = [
            (index, result) for index, result in enumerate(results) if result.get("score", 1.0) < 0.5
        ]
        if not items_to_recheck:
            recheck_status = "not_needed"
            print(f"No items to re-check for {dataset}.")
        elif args.skip_llm_recheck:
            recheck_status = "skipped_by_request"
            print(f"Skipping LLM recheck for {len(items_to_recheck)} incorrect items in {dataset}.")
        else:
            if judge_client is None:
                judge_client = create_judge_client()
            print(f"Re-checking {len(items_to_recheck)} items for {dataset}.")
            failures = []
            futures_to_index = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for index, item in items_to_recheck:
                    response_for_grading = item.get("student_response_for_grading", item["response"])
                    future = executor.submit(
                        process_example, judge_client, args.judge_model, item["answer"], response_for_grading
                    )
                    futures_to_index[future] = index
                for future in tqdm(
                    concurrent.futures.as_completed(futures_to_index),
                    total=len(items_to_recheck),
                    desc=f"Checking {dataset}",
                ):
                    original_index = futures_to_index[future]
                    try:
                        judged = future.result()
                        normalized_judgement = normalize_judgement(judged)
                    except Exception as error:
                        failures.append(f"index {original_index}: {error}")
                        continue
                    results[original_index]["llm_recheck_judgement"] = normalized_judgement
                    if normalized_judgement == "yes":
                        results[original_index]["score"] = 1
            if failures:
                raise RuntimeError(
                    f"LLM recheck failed for {len(failures)} item(s) in {dataset}: {failures[0]}"
                )
            recheck_status = "completed"

        for result in results:
            result["llm_recheck_status"] = recheck_status
        per_item_output = write_rechecked_results(source_path, results, overwrite_recheck)
        print(f"Saved per-item rechecked results to {per_item_output}")
        final_score = round(sum(result.get("score", 0) for result in results) / len(results) * 100, 2)
        print(f"Final score for {dataset}: {final_score}")
        result_entry = {
            "model": args.model_name,
            "dataset": dataset,
            "score": final_score,
            "larger_model": args.larger_model,
            "interaction_policy": args.interaction_policy,
            "collection_mode": args.collection_mode if args.interaction_policy == "hsp" else None,
            "output_tag": args.output_tag,
            "llm_recheck_status": recheck_status,
            "per_item_results": str(per_item_output),
        }
        new_results.append(result_entry)
        summary_output = write_rechecked_summary(per_item_output, result_entry, overwrite_recheck)
        print(f"Saved rechecked summary to {summary_output}")

    print("\n--- Re-checking complete ---")
    print(json.dumps(new_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
