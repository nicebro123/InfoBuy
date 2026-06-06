"""Fetch the reviewed NuminaMath source pool for HSP protocol cold start.

The first HSP protocol seed intentionally uses only the ``synthetic_math``
category from NuminaMath-CoT. Categories directly named after held-out
benchmarks or competition sets are not included in this source pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import socket
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DATASET_ID = "AI-MO/NuminaMath-CoT"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DATASET_URL = "https://huggingface.co/datasets/AI-MO/NuminaMath-CoT"
DATASET_LICENSE = "apache-2.0"
DEFAULT_SOURCE_CATEGORY = "synthetic_math"
EXCLUDED_SOURCE_CATEGORIES = [
    "math",
    "gsm8k",
    "synthetic_amc",
    "amc_aime",
    "olympiads",
]
VIEWER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
VIEWER_SIZE_URL = "https://datasets-server.huggingface.co/size"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a provenance-tracked NuminaMath source pool for HSP protocol SFT."
    )
    parser.add_argument("--output", required=True, help="Output raw source JSONL path.")
    parser.add_argument("--manifest", help="Optional JSON manifest path; defaults beside the output.")
    parser.add_argument("--version", default="v0_protocol_seed", help="Artifact version stored in the manifest.")
    parser.add_argument("--max_records", type=int, default=1000, help="Number of usable base problems.")
    parser.add_argument("--page_size", type=int, default=100, help="Dataset Viewer page size, at most 100.")
    parser.add_argument(
        "--request_interval_seconds",
        type=float,
        default=2.0,
        help="Delay between Viewer pages to avoid API rate limiting.",
    )
    parser.add_argument("--max_retries", type=int, default=6, help="Maximum retries for transient Viewer failures.")
    parser.add_argument(
        "--sampling_mode",
        choices=("prefix", "shuffled_pages"),
        default="prefix",
        help="Use prefix for a smoke pool, or shuffled_pages for a non-prefix training pool.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffled_pages sampling.")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint JSON path; shuffled_pages defaults to <output>.checkpoint.json.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint.")
    parser.add_argument(
        "--progress_every_pages",
        type=int,
        default=25,
        help="Print collection progress after this many completed pages; set to 0 to disable.",
    )
    parser.add_argument(
        "--source_category",
        default=DEFAULT_SOURCE_CATEGORY,
        help="NuminaMath source category to collect.",
    )
    parser.add_argument(
        "--allow_unreviewed_source",
        action="store_true",
        help="Permit a source category other than the reviewed synthetic_math seed category.",
    )
    return parser.parse_args()


def _get_url_json(
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


def _get_json(
    params: dict[str, Any],
    opener: Callable[..., Any] = urllib.request.urlopen,
    max_retries: int = 6,
) -> dict[str, Any]:
    return _get_url_json(
        VIEWER_ROWS_URL + "?" + urllib.parse.urlencode(params),
        opener,
        max_retries,
    )


def get_split_size(
    opener: Callable[..., Any] = urllib.request.urlopen,
    max_retries: int = 6,
) -> int:
    payload = _get_url_json(
        VIEWER_SIZE_URL + "?" + urllib.parse.urlencode({"dataset": DATASET_ID}),
        opener,
        max_retries,
    )
    for split in payload.get("size", {}).get("splits", []):
        if split.get("config") == DATASET_CONFIG and split.get("split") == DATASET_SPLIT:
            return int(split["num_rows"])
    raise ValueError(f"Could not resolve {DATASET_ID}/{DATASET_CONFIG}/{DATASET_SPLIT} row count.")


def _problem_fingerprint(problem: str) -> str:
    normalized = unicodedata.normalize("NFKC", problem).casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*([{}()[\],.=+\-*/^])\s*", r"\1", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _checkpoint_records_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(checkpoint_path.stem + ".records.jsonl")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    temporary_path.replace(path)


def _load_partial_records(path: Path, expected_count: int) -> list[dict[str, Any]]:
    if not path.exists():
        if expected_count:
            raise ValueError(f"Checkpoint references missing partial records file: {path}.")
        return []
    with path.open("r", encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    if len(records) < expected_count:
        raise ValueError(
            f"Checkpoint expected {expected_count} collected records, but {path} contains {len(records)}."
        )
    committed = records[:expected_count]
    with path.open("w", encoding="utf-8") as output:
        for record in committed:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return committed


def fetch_records(
    source_category: str,
    max_records: int,
    page_size: int,
    opener: Callable[..., Any] = urllib.request.urlopen,
    request_interval_seconds: float = 0.0,
    max_retries: int = 6,
    sampling_mode: str = "prefix",
    seed: int = 42,
    total_rows: int | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    progress_every_pages: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    if max_records <= 0:
        raise ValueError("--max_records must be greater than zero.")
    if not 1 <= page_size <= 100:
        raise ValueError("--page_size must be in the range [1, 100].")
    if request_interval_seconds < 0:
        raise ValueError("--request_interval_seconds must not be negative.")
    if max_retries < 0:
        raise ValueError("--max_retries must not be negative.")
    if sampling_mode not in {"prefix", "shuffled_pages"}:
        raise ValueError("--sampling_mode must be prefix or shuffled_pages.")
    if progress_every_pages < 0:
        raise ValueError("--progress_every_pages must not be negative.")
    if resume and checkpoint_path is None:
        raise ValueError("--resume requires a checkpoint path.")
    offsets: list[int] | None = None
    if sampling_mode == "shuffled_pages":
        if total_rows is None or total_rows <= 0:
            raise ValueError("shuffled_pages sampling requires a positive total_rows value.")
        offsets = list(range(0, total_rows, page_size))
        random.Random(seed).shuffle(offsets)

    request = {
        "source_category": source_category,
        "max_records": max_records,
        "page_size": page_size,
        "sampling_mode": sampling_mode,
        "seed": seed,
        "total_rows": total_rows,
    }
    records: list[dict[str, Any]] = []
    offset = 0
    page_index = 0
    scanned = 0
    pages_completed = 0
    partial_path = _checkpoint_records_path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path and resume:
        with checkpoint_path.open("r", encoding="utf-8") as source:
            checkpoint = json.load(source)
        if checkpoint.get("request") != request:
            raise ValueError("Checkpoint request parameters do not match the current collection request.")
        records = _load_partial_records(partial_path, int(checkpoint["records_written"]))
        offset = int(checkpoint["next_offset"])
        page_index = int(checkpoint["next_page_index"])
        scanned = int(checkpoint["rows_scanned"])
        pages_completed = int(checkpoint["pages_completed"])
    elif checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text("", encoding="utf-8")

    seen_problems: set[str] = {_problem_fingerprint(record["question"]) for record in records}

    def save_checkpoint(new_records: list[dict[str, Any]], status: str) -> None:
        if not checkpoint_path or not partial_path:
            return
        if new_records:
            with partial_path.open("a", encoding="utf-8") as output:
                for record in new_records:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
        _write_json_atomic(
            checkpoint_path,
            {
                "artifact_type": "hsp_source_fetch_checkpoint",
                "status": status,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "request": request,
                "partial_records_path": str(partial_path.resolve()),
                "records_written": len(records),
                "rows_scanned": scanned,
                "pages_completed": pages_completed,
                "next_offset": offset,
                "next_page_index": page_index,
            },
        )

    if checkpoint_path and not resume:
        save_checkpoint([], "in_progress")

    while len(records) < max_records:
        request_offset = offsets[page_index] if offsets is not None else offset
        payload = _get_json(
            {
                "dataset": DATASET_ID,
                "config": DATASET_CONFIG,
                "split": DATASET_SPLIT,
                "offset": request_offset,
                "length": page_size,
            },
            opener,
            max_retries,
        )
        rows = payload.get("rows", [])
        if not rows:
            break
        new_records: list[dict[str, Any]] = []
        for local_index, wrapped_row in enumerate(rows):
            row = wrapped_row.get("row", wrapped_row)
            scanned += 1
            if row.get("source") != source_category:
                continue
            question = str(row.get("problem", "")).strip()
            solution = str(row.get("solution", "")).strip()
            if not question or not solution:
                continue
            fingerprint = _problem_fingerprint(question)
            if fingerprint in seen_problems:
                continue
            seen_problems.add(fingerprint)
            row_index = int(wrapped_row.get("row_idx", request_offset + local_index))
            record = {
                "id": f"numinamath_cot_{source_category}_{row_index:07d}",
                "question": question,
                "gold_solution": solution,
                "source_dataset": DATASET_ID,
                "source_config": DATASET_CONFIG,
                "source_split": DATASET_SPLIT,
                "source_category": source_category,
                "source_row_index": row_index,
                "source_license": DATASET_LICENSE,
                "source_url": DATASET_URL,
            }
            records.append(record)
            new_records.append(record)
            if len(records) >= max_records:
                break
        if offsets is None:
            offset += len(rows)
            total = payload.get("num_rows_total")
            if len(rows) < page_size or (isinstance(total, int) and offset >= total):
                break
        else:
            page_index += 1
        pages_completed += 1
        save_checkpoint(new_records, "in_progress")
        if progress_every_pages and pages_completed % progress_every_pages == 0:
            print(
                f"Scanned {scanned} rows across {pages_completed} pages; "
                f"collected {len(records)}/{max_records}.",
                flush=True,
            )
        if offsets is not None and page_index >= len(offsets):
            break
        if len(records) < max_records and request_interval_seconds:
            time.sleep(request_interval_seconds)

    if len(records) < max_records:
        raise ValueError(
            f"Only collected {len(records)} unique records from source={source_category!r}; "
            f"requested {max_records}."
        )
    save_checkpoint([], "complete")
    return records, scanned


def build_manifest(
    output_path: Path,
    records: list[dict[str, Any]],
    scanned: int,
    source_category: str,
    version: str = "v0_protocol_seed",
    sampling_mode: str = "prefix",
    seed: int = 42,
) -> dict[str, Any]:
    return {
        "artifact_type": "hsp_raw_problem_pool",
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": DATASET_ID,
        "source_url": DATASET_URL,
        "source_license": DATASET_LICENSE,
        "source_config": DATASET_CONFIG,
        "source_split": DATASET_SPLIT,
        "selection": {
            "included_source_category": source_category,
            "sampling_mode": sampling_mode,
            "sampling_seed": seed if sampling_mode == "shuffled_pages" else None,
            "sampling_description": (
                "Deterministically shuffled Dataset Viewer pages from the full train split; "
                "kept matching source-category records until the requested pool size was met."
                if sampling_mode == "shuffled_pages"
                else "Prefix scan of the train split for smoke/pilot use only."
            ),
            "excluded_direct_benchmark_or_competition_categories": EXCLUDED_SOURCE_CATEGORIES,
            "decontamination_status": (
                "Source-category isolation completed; exact text/near-duplicate comparison "
                "against held-out evaluation sets remains required before final experiments."
            ),
        },
        "rows_scanned": scanned,
        "records_written": len(records),
        "output_path": str(output_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    if (
        args.source_category != DEFAULT_SOURCE_CATEGORY
        and not args.allow_unreviewed_source
    ):
        raise ValueError(
            "Only source_category=synthetic_math is reviewed for the v0 seed. "
            "Pass --allow_unreviewed_source only after documenting a new source decision."
        )
    output_path = Path(args.output)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_path.with_name(output_path.stem + ".manifest.json")
    )
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else (
            output_path.with_suffix(output_path.suffix + ".checkpoint.json")
            if args.sampling_mode == "shuffled_pages"
            else None
        )
    )
    total_rows = get_split_size(max_retries=args.max_retries) if args.sampling_mode == "shuffled_pages" else None
    records, scanned = fetch_records(
        args.source_category,
        args.max_records,
        args.page_size,
        request_interval_seconds=args.request_interval_seconds,
        max_retries=args.max_retries,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
        total_rows=total_rows,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
        progress_every_pages=args.progress_every_pages,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as output:
        json.dump(
            build_manifest(
                output_path,
                records,
                scanned,
                args.source_category,
                args.version,
                args.sampling_mode,
                args.seed,
            ),
            output,
            ensure_ascii=False,
            indent=2,
        )
        output.write("\n")
    print(f"Wrote {len(records)} source records to {output_path}; scanned {scanned} rows.")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
