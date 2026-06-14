#!/usr/bin/env python3
"""Run a tiny HSP rollout smoke and validate interaction events."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def default_model_path() -> str:
    checkpoint_root = os.environ.get("INFOBUY_CKPT")
    if checkpoint_root:
        return str(Path(checkpoint_root) / "sft" / "qwen3-0.6b-hsp-sft")
    return "checkpoints/sft/qwen3-0.6b-hsp-sft"


def default_data_path() -> str:
    generated = os.environ.get("INFOBUY_GENERATED_DATA")
    if generated:
        return str(Path(generated) / "raw" / "numinamath_cot_synthetic_math_validation_smoke.jsonl")
    return "datasets/infobuy/raw/numinamath_cot_synthetic_math_validation_smoke.jsonl"


def default_storage_path() -> str:
    return os.environ.get("STORAGE_PATH", ".")


def safe_output_component(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return rendered[:64] or "run"


def result_dataset_key(dataset: str, dataset_name: str | None, output_tag: str | None) -> str:
    parts = [dataset]
    if dataset_name:
        identity = str(Path(dataset_name).expanduser().resolve()) if dataset == "local_json" else dataset_name
        readable = Path(dataset_name).stem if dataset == "local_json" else dataset_name.rsplit("/", 1)[-1]
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
        parts.extend([safe_output_component(readable), digest])
    if output_tag:
        parts.append(safe_output_component(output_tag))
    return "_".join(parts)


def result_output_path(
    storage_path: str,
    small_model: str,
    larger_model: str,
    dataset_name: str,
    output_tag: str,
    collection_mode: str,
) -> Path:
    output_dir = Path(storage_path) / "evaluation" / f"{small_model.replace('/', '_')}_{larger_model}"
    dataset_key = result_dataset_key("local_json", dataset_name, output_tag)
    suffix = "_hsp" if collection_mode == "policy" else f"_hsp_{collection_mode}"
    return output_dir / f"results_{dataset_key}{suffix}.json"


def check_teacher(url: str, timeout: float) -> None:
    payload = json.dumps([{"prompt": "What is 2+2?", "max_tokens": 8}]).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status >= 400:
                raise RuntimeError(f"Teacher health check failed with HTTP {response.status}.")
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Teacher health check failed for {url}: {error}") from error
    if "results" not in data:
        raise RuntimeError(f"Teacher health check response did not contain results: {data}")


def validate_result_file(path: Path, collection_mode: str) -> dict[str, Any]:
    from SFT_stage.preflight_hsp import validate_dataset

    if not path.exists():
        raise FileNotFoundError(f"Expected rollout result was not created: {path}")
    with path.open("r", encoding="utf-8") as stream:
        items = json.load(stream)
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} must contain a non-empty JSON list.")

    errors = []
    ask_count = 0
    verify_count = 0
    accept_count = 0
    teacher_tokens_used = 0
    protocol_examples = []
    for index, item in enumerate(items):
        if item.get("interaction_policy") != "hsp":
            errors.append(f"item[{index}] is not an HSP result.")
        if item.get("collection_error"):
            errors.append(f"item[{index}] collection_error={item['collection_error']!r}.")
        ask_count += int(item.get("ask_count", 0))
        verify_count += int(item.get("verify_count", 0))
        accept_count += int(item.get("accept_count", 0))
        teacher_tokens_used += int(item.get("teacher_tokens_used", 0))
        segments = item.get("segments")
        if isinstance(segments, list):
            protocol_examples.append({"segments": segments})
        else:
            errors.append(f"item[{index}] is missing segments.")

    if collection_mode == "force_ask_first" and ask_count < len(items):
        errors.append(f"force_ask_first expected at least one ask per item; got ask_count={ask_count}.")
    if collection_mode == "force_verify_after_draft" and verify_count < len(items):
        errors.append(f"force_verify_after_draft expected at least one verify per item; got verify_count={verify_count}.")
    if collection_mode in {"force_ask_first", "force_verify_after_draft"} and teacher_tokens_used <= 0:
        errors.append(f"{collection_mode} did not record teacher token usage.")

    if protocol_examples:
        protocol_report = validate_dataset(protocol_examples)
        if protocol_report["errors"]:
            errors.extend(f"protocol: {error}" for error in protocol_report["errors"][:10])

    if errors:
        raise ValueError("; ".join(errors))
    return {
        "path": str(path),
        "items": len(items),
        "ask_count": ask_count,
        "verify_count": verify_count,
        "accept_count": accept_count,
        "teacher_tokens_used": teacher_tokens_used,
    }


def run_mode(args: argparse.Namespace, collection_mode: str, output_tag: str) -> dict[str, Any]:
    large_model_url = f"http://127.0.0.1:{args.port}/generate"
    command = [
        sys.executable,
        "-m",
        "eval.generate_withhelp",
        "--small_model",
        args.model_path,
        "--dataset",
        "local_json",
        "--name",
        args.data,
        "--larger_model",
        args.teacher_name,
        "--large_model_url",
        large_model_url,
        "--interaction_policy",
        "hsp",
        "--collection_mode",
        collection_mode,
        "--max_examples",
        str(args.max_examples),
        "--samples_per_question",
        str(args.samples_per_question),
        "--max_interactions",
        str(args.max_interactions),
        "--ask_budget_tokens",
        str(args.ask_budget_tokens),
        "--verify_budget_tokens",
        str(args.verify_budget_tokens),
        "--student_temperature",
        str(args.student_temperature),
        "--output_tag",
        output_tag,
        "--overwrite_results",
    ]
    env = dict(os.environ)
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    subprocess.run(command, cwd=args.repo_root, env=env, check=True)
    result_path = result_output_path(
        storage_path=args.storage_path,
        small_model=args.model_path,
        larger_model=args.teacher_name,
        dataset_name=args.data,
        output_tag=output_tag,
        collection_mode=collection_mode,
    )
    return validate_result_file(result_path, collection_mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and validate a tiny HSP rollout smoke.")
    parser.add_argument("--model_path", default=default_model_path())
    parser.add_argument("--data", default=default_data_path())
    parser.add_argument("--storage_path", default=default_storage_path())
    parser.add_argument("--teacher_name", default="Qwen3-8B")
    parser.add_argument("--port", type=int, default=7778)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--max_examples", type=int, default=1)
    parser.add_argument("--samples_per_question", type=int, default=1)
    parser.add_argument("--max_interactions", type=int, default=2)
    parser.add_argument("--ask_budget_tokens", type=int, default=48)
    parser.add_argument("--verify_budget_tokens", type=int, default=64)
    parser.add_argument("--student_temperature", type=float, default=0.7)
    parser.add_argument(
        "--collection_modes",
        default="force_ask_first,force_verify_after_draft",
        help="Comma-separated modes. Use policy to include natural policy rollout.",
    )
    parser.add_argument("--output_tag", default=None)
    parser.add_argument("--teacher_check_timeout", type=float, default=10.0)
    parser.add_argument("--skip_teacher_check", action="store_true")
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_examples <= 0 or args.samples_per_question <= 0:
        raise ValueError("--max_examples and --samples_per_question must be positive.")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Rollout smoke data not found: {args.data}")
    modes = [mode.strip() for mode in args.collection_modes.split(",") if mode.strip()]
    if not modes:
        raise ValueError("--collection_modes must contain at least one mode.")
    if not args.skip_teacher_check:
        check_teacher(f"http://127.0.0.1:{args.port}/generate", args.teacher_check_timeout)

    base_tag = args.output_tag or f"rollout_smoke_{int(time.time())}"
    mode_reports = []
    for mode in modes:
        mode_tag = f"{base_tag}_{mode}"
        mode_reports.append(run_mode(args, mode, mode_tag))

    report = {
        "ok": True,
        "model_path": args.model_path,
        "data": args.data,
        "teacher_url": f"http://127.0.0.1:{args.port}/generate",
        "collection_modes": modes,
        "mode_reports": mode_reports,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if args.as_json:
        print(rendered)
    else:
        print("HSP rollout smoke: PASS")
        for mode_report in mode_reports:
            print(
                f"  {mode_report['path']}: items={mode_report['items']} "
                f"ask={mode_report['ask_count']} verify={mode_report['verify_count']} "
                f"accept={mode_report['accept_count']} teacher_tokens={mode_report['teacher_tokens_used']}"
            )
        if args.output:
            print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
