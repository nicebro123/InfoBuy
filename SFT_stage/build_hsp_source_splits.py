"""Split a NuminaMath HSP source pool after held-out benchmark filtering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import socket
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from SFT_stage.build_hsp_sft import normalize_record
except ImportError:
    from build_hsp_sft import normalize_record


VIEWER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
MATH500_URL = "https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv"
PROMPT_SUFFIX_PATTERN = re.compile(
    r"(?:please reason step by step|solve the following math problem step by step|"
    r"remember to put your answer|put your final answer|the last line of your response)",
    flags=re.IGNORECASE,
)
HELDOUT_HUB_SOURCES = (
    ("gsm8k", "openai/gsm8k", "main", "test", "question"),
    ("amc23", "zwhe99/amc23", "default", "test", "question"),
    ("minerva", "zwhe99/simplerl-minerva-math", "default", "test", "problem"),
    ("olympiad_bench", "zwhe99/simplerl-OlympiadBench", "default", "test", "question"),
    ("aime_2024", "HuggingFaceH4/aime_2024", "default", "train", "problem"),
    ("aime_2025", "yentinglin/aime_2025", "default", "train", "problem"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic, benchmark-filtered raw HSP train/validation source pools."
    )
    parser.add_argument("--input", required=True, help="Raw NuminaMath source JSONL.")
    parser.add_argument("--train_output", required=True, help="Output JSONL for train base problems.")
    parser.add_argument("--validation_output", required=True, help="Output JSONL for validation base problems.")
    parser.add_argument("--manifest", required=True, help="Output split/decontamination manifest JSON.")
    parser.add_argument(
        "--heldout_snapshot",
        help="Optional held-out question snapshot JSON path; defaults beside the manifest.",
    )
    parser.add_argument(
        "--heldout_snapshot_input",
        help="Reuse a previously written held-out snapshot instead of fetching current benchmark rows.",
    )
    parser.add_argument("--train_size", type=int, required=True)
    parser.add_argument("--validation_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--near_duplicate_threshold",
        type=float,
        default=0.85,
        help="Character 5-gram Jaccard threshold used to reject likely held-out duplicates.",
    )
    parser.add_argument("--page_size", type=int, default=100)
    parser.add_argument("--request_interval_seconds", type=float, default=0.25)
    parser.add_argument("--max_retries", type=int, default=6)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def normalize_question(text: str) -> str:
    text = re.sub(
        r"\\n(?=(?:please|solve|remember|the last line))",
        "\n",
        str(text),
        flags=re.IGNORECASE,
    )
    lines = [
        line
        for line in unicodedata.normalize("NFKC", text).splitlines()
        if not PROMPT_SUFFIX_PATTERN.search(line)
    ]
    compact = " ".join(lines).casefold()
    compact = re.sub(r"\s+", " ", compact)
    compact = re.sub(r"\s*([{}()[\],.=+\-*/^])\s*", r"\1", compact)
    return compact.strip()


def char_ngrams(text: str, size: int = 5) -> set[str]:
    if len(text) <= size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def _get_json(
    url: str,
    opener: Callable[..., Any] = urllib.request.urlopen,
    max_retries: int = 6,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            with opener(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt >= max_retries:
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            delay = float(retry_after) if retry_after else min(2 ** (attempt + 1), 60)
            time.sleep(delay)
        except (TimeoutError, socket.timeout, urllib.error.URLError):
            if attempt >= max_retries:
                raise
            time.sleep(min(2 ** (attempt + 1), 60))
    raise RuntimeError("Unreachable retry loop.")


def fetch_hub_questions(
    dataset: str,
    config: str,
    split: str,
    question_field: str,
    page_size: int,
    interval_seconds: float,
    max_retries: int,
) -> list[str]:
    questions: list[str] = []
    offset = 0
    while True:
        params = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        payload = _get_json(f"{VIEWER_ROWS_URL}?{params}", max_retries=max_retries)
        rows = payload.get("rows", [])
        if not rows:
            break
        questions.extend(
            str(wrapped.get("row", wrapped).get(question_field, "")).strip()
            for wrapped in rows
            if str(wrapped.get("row", wrapped).get(question_field, "")).strip()
        )
        offset += len(rows)
        total = payload.get("num_rows_total")
        if len(rows) < page_size or (isinstance(total, int) and offset >= total):
            break
        if interval_seconds:
            time.sleep(interval_seconds)
    return questions


def fetch_math500_questions(max_retries: int = 6) -> list[str]:
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(MATH500_URL, timeout=60) as response:
                text = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt >= max_retries:
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            delay = float(retry_after) if retry_after else min(2 ** (attempt + 1), 60)
            time.sleep(delay)
        except (TimeoutError, socket.timeout, urllib.error.URLError):
            if attempt >= max_retries:
                raise
            time.sleep(min(2 ** (attempt + 1), 60))
    return [row["Question"] for row in csv.DictReader(io.StringIO(text)) if row.get("Question")]


def fetch_heldout_questions(
    page_size: int,
    interval_seconds: float,
    max_retries: int,
) -> dict[str, list[str]]:
    questions = {"math_500": fetch_math500_questions(max_retries=max_retries)}
    for name, dataset, config, split, field in HELDOUT_HUB_SOURCES:
        questions[name] = fetch_hub_questions(
            dataset,
            config,
            split,
            field,
            page_size,
            interval_seconds,
            max_retries,
        )
    return questions


def prepare_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        normalized = normalize_record(record, index)
        if normalized is None:
            removed.append(
                {
                    "id": record.get("id"),
                    "reason": "missing_question_solution_or_answer",
                }
            )
        else:
            prepared.append(normalized)
    return prepared, removed


def find_contamination(
    records: Iterable[dict[str, Any]],
    heldout_by_source: dict[str, list[str]],
    near_duplicate_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("--near_duplicate_threshold must be in the range (0, 1].")

    heldout: list[tuple[str, str, set[str]]] = []
    exact_sources: defaultdict[str, set[str]] = defaultdict(set)
    gram_index: defaultdict[str, list[int]] = defaultdict(list)
    for source_name, questions in heldout_by_source.items():
        for question in questions:
            normalized = normalize_question(question)
            if not normalized:
                continue
            grams = char_ngrams(normalized)
            index = len(heldout)
            heldout.append((source_name, normalized, grams))
            exact_sources[normalized].add(source_name)
            for gram in grams:
                gram_index[gram].append(index)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for record in records:
        question = normalize_question(record.get("question", ""))
        if not question:
            removed.append({"id": record.get("id"), "reason": "empty_question"})
            continue
        if question in exact_sources:
            removed.append(
                {
                    "id": record.get("id"),
                    "reason": "exact_match",
                    "heldout_sources": sorted(exact_sources[question]),
                }
            )
            continue
        grams = char_ngrams(question)
        overlaps: Counter[int] = Counter()
        for gram in grams:
            overlaps.update(gram_index.get(gram, []))
        rejected = None
        for index, overlap in overlaps.most_common():
            source_name, _, heldout_grams = heldout[index]
            minimum_overlap = math.ceil(
                near_duplicate_threshold
                * (len(grams) + len(heldout_grams))
                / (1 + near_duplicate_threshold)
            )
            if overlap < minimum_overlap:
                continue
            similarity = overlap / len(grams.union(heldout_grams))
            if similarity >= near_duplicate_threshold:
                rejected = {
                    "id": record.get("id"),
                    "reason": "near_duplicate",
                    "heldout_sources": [source_name],
                    "similarity": round(similarity, 6),
                }
                break
        if rejected:
            removed.append(rejected)
        else:
            kept.append(record)
    return kept, removed


def deduplicate_source_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    ordered = sorted(records, key=lambda record: str(record.get("id", "")))
    for record in ordered:
        fingerprint = normalize_question(record.get("question", ""))
        original_id = seen.get(fingerprint)
        if original_id is not None:
            removed.append(
                {
                    "id": record.get("id"),
                    "reason": "internal_normalized_duplicate",
                    "duplicate_of": original_id,
                }
            )
            continue
        seen[fingerprint] = str(record.get("id"))
        kept.append(record)
    return kept, removed


def deterministic_split(
    records: list[dict[str, Any]],
    train_size: int,
    validation_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_size <= 0 or validation_size <= 0:
        raise ValueError("Train and validation sizes must be greater than zero.")
    required = train_size + validation_size
    if len(records) < required:
        raise ValueError(f"Only {len(records)} clean records remain; {required} are required.")

    def key(record: dict[str, Any]) -> str:
        identifier = str(record.get("id") or normalize_question(record.get("question", "")))
        return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()

    ordered = sorted(records, key=key)
    return ordered[:train_size], ordered[train_size:required]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_heldout_snapshot(
    path: Path,
    heldout: dict[str, list[str]],
) -> dict[str, Any]:
    normalized_by_source = {
        name: sorted({normalize_question(question) for question in questions if normalize_question(question)})
        for name, questions in heldout.items()
    }
    hashes = {
        name: hashlib.sha256(
            "\n".join(questions).encode("utf-8")
        ).hexdigest()
        for name, questions in normalized_by_source.items()
    }
    combined_payload = json.dumps(normalized_by_source, ensure_ascii=False, sort_keys=True)
    snapshot = {
        "artifact_type": "hsp_heldout_snapshot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "normalization": "NFKC, casefold, whitespace/punctuation normalization, instruction-line removal",
        "question_counts": {name: len(items) for name, items in heldout.items()},
        "normalized_unique_counts": {name: len(items) for name, items in normalized_by_source.items()},
        "normalized_sha256": hashes,
        "combined_normalized_sha256": hashlib.sha256(combined_payload.encode("utf-8")).hexdigest(),
        "questions": heldout,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(snapshot, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return snapshot


def load_heldout_snapshot(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as source:
        snapshot = json.load(source)
    questions = snapshot.get("questions")
    if not isinstance(questions, dict) or not all(
        isinstance(name, str) and isinstance(items, list) and all(isinstance(item, str) for item in items)
        for name, items in questions.items()
    ):
        raise ValueError(f"Invalid held-out snapshot questions in {path}.")
    return questions


def main() -> None:
    args = parse_args()
    raw_records = load_jsonl(Path(args.input))
    prepared_records, invalid_records = prepare_records(raw_records)
    heldout = (
        load_heldout_snapshot(Path(args.heldout_snapshot_input))
        if args.heldout_snapshot_input
        else fetch_heldout_questions(
            args.page_size,
            args.request_interval_seconds,
            args.max_retries,
        )
    )
    clean_records, heldout_removed = find_contamination(
        prepared_records,
        heldout,
        args.near_duplicate_threshold,
    )
    unique_records, internal_duplicates = deduplicate_source_records(clean_records)
    train, validation = deterministic_split(
        unique_records,
        args.train_size,
        args.validation_size,
        args.seed,
    )
    train_path = Path(args.train_output)
    validation_path = Path(args.validation_output)
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    manifest_path = Path(args.manifest)
    snapshot_path = (
        Path(args.heldout_snapshot)
        if args.heldout_snapshot
        else manifest_path.with_name(manifest_path.stem + ".heldout_snapshot.json")
    )
    heldout_snapshot = write_heldout_snapshot(snapshot_path, heldout)
    removed = invalid_records + heldout_removed + internal_duplicates
    manifest = {
        "artifact_type": "hsp_source_split",
        "version": "v1_numinamath_mainline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(args.input).resolve()),
        "input_records": len(raw_records),
        "prepared_records_with_gold_answer": len(prepared_records),
        "source_dataset": "AI-MO/NuminaMath-CoT",
        "source_category": "synthetic_math",
        "heldout_profile": "relayllm_math_benchmarks_plus_amc",
        "heldout_question_counts": {name: len(items) for name, items in heldout.items()},
        "heldout_snapshot": {
            "path": str(snapshot_path.resolve()),
            "input_path": str(Path(args.heldout_snapshot_input).resolve()) if args.heldout_snapshot_input else None,
            "combined_normalized_sha256": heldout_snapshot["combined_normalized_sha256"],
            "normalized_sha256": heldout_snapshot["normalized_sha256"],
        },
        "decontamination": {
            "normalization": "NFKC, casefold, whitespace/punctuation normalization, instruction-line removal",
            "exact_match_removed": sum(item["reason"] == "exact_match" for item in removed),
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "near_duplicate_removed": sum(item["reason"] == "near_duplicate" for item in removed),
            "internal_normalized_duplicate_removed": sum(
                item["reason"] == "internal_normalized_duplicate" for item in removed
            ),
            "invalid_record_removed": sum(
                item["reason"] == "missing_question_solution_or_answer" for item in removed
            ),
            "removed_records": removed,
        },
        "split": {
            "method": (
                "normalize and deduplicate equivalent source questions, then "
                "sha256(seed:source_id) deterministic split before protocol expansion"
            ),
            "seed": args.seed,
            "clean_records_available": len(unique_records),
            "train_records": len(train),
            "validation_records": len(validation),
            "train_output": str(train_path.resolve()),
            "validation_output": str(validation_path.resolve()),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"Read {len(raw_records)} records; removed {len(removed)} invalid, held-out, or duplicate records.")
    print(f"Wrote train={len(train)} to {train_path}; validation={len(validation)} to {validation_path}.")
    print(f"Held-out snapshot: {snapshot_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
