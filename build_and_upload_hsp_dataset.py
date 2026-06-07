#!/usr/bin/env python3
"""Build HSP SFT datasets and optionally upload to HuggingFace Hub.

The uploaded dataset uses a flat question/answer/system SFT schema:
    question (str):  The math problem (with prompt suffix)
    answer (str):    The complete response (student text + teacher context interleaved)
    system (str):    System prompt (empty string)

This keeps the data compatible with standard SFTTrainer pipelines (e.g. SFT_stage/train_hsp.py).
An additional metadata column 'sample_type' is included for filtering.

Usage:
    # Step 1: Build only (inspect locally first)
    python build_and_upload_hsp_dataset.py --build_only

    # Step 2: Upload after review
    python build_and_upload_hsp_dataset.py --upload_only \
        --hf_repo_id YOUR_USERNAME/hsp-protocol-sft

    # Or do both in one go
    python build_and_upload_hsp_dataset.py \
        --hf_repo_id YOUR_USERNAME/hsp-protocol-sft

    # Custom source pool size (default 1000)
    python build_and_upload_hsp_dataset.py --build_only --max_source_records 2000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def default_generated_data_dir() -> Path:
    configured = os.environ.get("INFOBUY_GENERATED_DATA") or os.environ.get("HSP_GENERATED_DATA")
    if configured:
        return Path(configured).expanduser()
    store_root = Path(os.environ.get("INFOBUY_STORE", REPO_ROOT.parent / "InfoBuy_store")).expanduser()
    return store_root / "datasets" / "infobuy"


def configure_data_dir(data_dir: str | Path) -> None:
    global DATA_DIR, RAW_DIR, PROTOCOL_DIR, MANIFEST_DIR, FLAT_DIR
    global DEFAULT_TRAIN_RAW, DEFAULT_VAL_RAW, DEFAULT_SPLIT_MANIFEST
    global DEFAULT_TRAIN_PROTOCOL, DEFAULT_VAL_PROTOCOL, DEFAULT_TRAIN_FLAT, DEFAULT_VAL_FLAT

    DATA_DIR = Path(data_dir).expanduser()
    RAW_DIR = DATA_DIR / "raw"
    PROTOCOL_DIR = DATA_DIR / "protocol"
    MANIFEST_DIR = DATA_DIR / "manifests"
    FLAT_DIR = DATA_DIR / "flat"  # RelayLLM-compatible flat format

    DEFAULT_TRAIN_RAW = RAW_DIR / "numinamath_cot_synthetic_math_train_pilot_v1_800.jsonl"
    DEFAULT_VAL_RAW = RAW_DIR / "numinamath_cot_synthetic_math_validation_pilot_v1_200.jsonl"
    DEFAULT_SPLIT_MANIFEST = MANIFEST_DIR / "numinamath_cot_synthetic_math_split_pilot_v1.manifest.json"
    DEFAULT_TRAIN_PROTOCOL = PROTOCOL_DIR / "hsp_protocol_train_pilot_v1.jsonl"
    DEFAULT_VAL_PROTOCOL = PROTOCOL_DIR / "hsp_protocol_validation_pilot_v1.jsonl"
    DEFAULT_TRAIN_FLAT = FLAT_DIR / "hsp_sft_train.jsonl"
    DEFAULT_VAL_FLAT = FLAT_DIR / "hsp_sft_validation.jsonl"


configure_data_dir(default_generated_data_dir())


def run_command(cmd: list[str], description: str) -> None:
    """Run a command and raise on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {description}")


# ---------------------------------------------------------------------------
# Build steps 1-4: fetch, split, generate protocol data, validate
# ---------------------------------------------------------------------------

def step1_fetch_source(max_records: int, force: bool) -> None:
    """Fetch NuminaMath-CoT synthetic_math source pool."""
    output = RAW_DIR / f"numinamath_cot_synthetic_math_seed_v0_{max_records}.jsonl"
    manifest = MANIFEST_DIR / f"numinamath_cot_synthetic_math_seed_v0_{max_records}.manifest.json"

    if output.exists() and not force:
        count = sum(1 for _ in output.open())
        print(f"\n[Step 1] Source pool already exists: {output} ({count} records)")
        print("         Use --force to re-fetch.")
        return

    run_command(
        [
            sys.executable, "-m", "SFT_stage.fetch_hsp_source_dataset",
            "--max_records", str(max_records),
            "--request_interval_seconds", "2",
            "--output", str(output),
            "--manifest", str(manifest),
        ],
        f"Fetching {max_records} problems from NuminaMath-CoT/synthetic_math",
    )


def step2_split_and_decontaminate(
    max_records: int, train_size: int, val_size: int, seed: int, force: bool,
) -> None:
    """Split source pool into train/val with benchmark decontamination."""
    source = RAW_DIR / f"numinamath_cot_synthetic_math_seed_v0_{max_records}.jsonl"
    train_out = RAW_DIR / f"numinamath_cot_synthetic_math_train_pilot_v1_{train_size}.jsonl"
    val_out = RAW_DIR / f"numinamath_cot_synthetic_math_validation_pilot_v1_{val_size}.jsonl"
    manifest = MANIFEST_DIR / "numinamath_cot_synthetic_math_split_pilot_v1.manifest.json"

    if not source.exists():
        raise FileNotFoundError(f"Source pool not found: {source}. Run Step 1 first.")

    if train_out.exists() and val_out.exists() and not force:
        tc = sum(1 for _ in train_out.open())
        vc = sum(1 for _ in val_out.open())
        print(f"\n[Step 2] Split data already exists: train={tc}, val={vc}")
        print("         Use --force to rebuild.")
        return

    snapshot_path = MANIFEST_DIR / "numinamath_cot_synthetic_math_split_pilot_v1.manifest.heldout_snapshot.json"
    extra_args = ["--heldout_snapshot_input", str(snapshot_path)] if snapshot_path.exists() else []

    run_command(
        [
            sys.executable, "-m", "SFT_stage.build_hsp_source_splits",
            "--input", str(source),
            "--train_size", str(train_size),
            "--validation_size", str(val_size),
            "--seed", str(seed),
            "--train_output", str(train_out),
            "--validation_output", str(val_out),
            "--manifest", str(manifest),
        ] + extra_args,
        f"Splitting into train={train_size} / val={val_size} with decontamination",
    )


def step2b_calibrate_difficulty(
    train_size: int,
    val_size: int,
    model: str,
    samples_k: int,
    mode: str,
    target_accuracy: float,
    force: bool,
) -> tuple[Path, Path]:
    """Optional: tag/select train/val pools by difficulty (needs GPU).

    For SFT cold-start (teaching the protocol tokens) the default mode is ``tag``:
    keep ALL problems -- easy AND hard -- and just annotate solve_rate + report the
    difficulty histogram. The ``target``/``stratify`` modes are for narrowing or
    balancing (e.g. RL-stage data). Returns the calibrated train/val paths, which
    Step 3 then builds protocol data from.
    """
    raw_train = RAW_DIR / f"numinamath_cot_synthetic_math_train_pilot_v1_{train_size}.jsonl"
    raw_val = RAW_DIR / f"numinamath_cot_synthetic_math_validation_pilot_v1_{val_size}.jsonl"
    cal_train = RAW_DIR / f"numinamath_cot_synthetic_math_train_calibrated_{train_size}.jsonl"
    cal_val = RAW_DIR / f"numinamath_cot_synthetic_math_validation_calibrated_{val_size}.jsonl"

    if not raw_train.exists() or not raw_val.exists():
        raise FileNotFoundError("Raw train/val data not found. Run Step 2 first.")

    if cal_train.exists() and cal_val.exists() and not force:
        print(f"\n[Step 2b] Calibrated data already exists: {cal_train.name}, {cal_val.name}")
        print("          Use --force to recalibrate.")
        return cal_train, cal_val

    for raw_path, cal_path, label in [(raw_train, cal_train, "train"), (raw_val, cal_val, "validation")]:
        report_path = MANIFEST_DIR / f"difficulty_{label}.json"
        cmd = [sys.executable, "-m", "SFT_stage.calibrate_hsp_difficulty",
               "--input", str(raw_path), "--output", str(cal_path),
               "--report", str(report_path),
               "--model_path", model, "--samples_k", str(samples_k),
               "--mode", mode]
        if mode == "target":
            cmd += ["--target_accuracy", str(target_accuracy)]
        run_command(cmd, f"Calibrating {label} difficulty with {model} (mode={mode})")
    return cal_train, cal_val


def step3_build_protocol_sft(
    train_size: int,
    val_size: int,
    seed: int,
    force: bool,
    train_raw_override: Path | None = None,
    val_raw_override: Path | None = None,
) -> None:
    """Generate structured HSP protocol SFT data."""
    train_raw = train_raw_override or (RAW_DIR / f"numinamath_cot_synthetic_math_train_pilot_v1_{train_size}.jsonl")
    val_raw = val_raw_override or (RAW_DIR / f"numinamath_cot_synthetic_math_validation_pilot_v1_{val_size}.jsonl")
    train_out = DEFAULT_TRAIN_PROTOCOL
    val_out = DEFAULT_VAL_PROTOCOL

    if not train_raw.exists() or not val_raw.exists():
        raise FileNotFoundError("Raw train/val data not found. Run Step 2 first.")

    if train_out.exists() and val_out.exists() and not force:
        tc = sum(1 for _ in train_out.open())
        vc = sum(1 for _ in val_out.open())
        print(f"\n[Step 3] Protocol SFT data already exists: train={tc}, val={vc}")
        print("         Use --force to rebuild.")
        return

    run_command(
        [sys.executable, "-m", "SFT_stage.build_hsp_sft",
         "--input", str(train_raw), "--output", str(train_out),
         "--variants_per_problem", "1", "--seed", str(seed)],
        "Building protocol SFT training data (weighted sampling, 1 variant/problem)",
    )
    run_command(
        [sys.executable, "-m", "SFT_stage.build_hsp_sft",
         "--input", str(val_raw), "--output", str(val_out),
         "--emit_all_types", "--seed", str(seed)],
        "Building protocol SFT validation data (all 6 types per problem)",
    )


def step4_validate() -> None:
    """Validate protocol data integrity."""
    for path, label in [(DEFAULT_TRAIN_PROTOCOL, "train"), (DEFAULT_VAL_PROTOCOL, "validation")]:
        if not path.exists():
            raise FileNotFoundError(f"Protocol {label} data not found: {path}.")
        run_command(
            [sys.executable, "-m", "SFT_stage.preflight_hsp",
             "--dataset", str(path), "--require_all_types", "--require_context_tokens"],
            f"Validating {label} protocol data",
        )


# ---------------------------------------------------------------------------
# Step 5: Convert to RelayLLM-compatible flat format (question / answer / system)
# ---------------------------------------------------------------------------

def segments_to_answer(segments: list[dict]) -> str:
    """Concatenate all non-user segments into a single answer string.

    Everything after the user prompt becomes a single flat 'answer' string, matching
    the RelayLLM-compatible question/answer/system SFT schema.

    For HSP data, the answer includes student text (with <ASK>N</ASK> / <VERIFY>N</VERIFY>
    / <ACCEPT>) interleaved with teacher context (wrapped in <TEACHER_HELP>/</TEACHER_HELP>
    or <TEACHER_REVIEW>/</TEACHER_REVIEW>).
    """
    parts: list[str] = []
    for seg in segments:
        if seg.get("source") in ("system", "user"):
            continue  # skip — this goes into the 'question' field
        parts.append(seg.get("text", ""))
    return "".join(parts)


def extract_question(segments: list[dict]) -> str:
    """Extract the user question from the first user segment."""
    for seg in segments:
        if seg.get("source") == "user":
            return seg.get("text", "")
    return ""


def convert_to_flat_format(
    protocol_path: Path,
    flat_path: Path,
    force: bool,
) -> None:
    """Convert structured HSP protocol data to RelayLLM-compatible flat format.

    Output format (identical to HINT-lab/sft_Qwen_Qwen3-0.6B):
        question (str):     User prompt (already includes "Please reason step by step...")
        answer (str):       Student + teacher interleaved text
        system (str):       Empty string
        sample_type (str):  HSP sample type (for metadata/filtering)
        gold_answer (str):  Ground truth answer
    """
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol data not found: {protocol_path}")

    if flat_path.exists() and not force:
        count = sum(1 for _ in flat_path.open())
        print(f"\n  Flat data already exists: {flat_path} ({count} examples)")
        print("  Use --force to rebuild.")
        return

    flat_path.parent.mkdir(parents=True, exist_ok=True)

    with protocol_path.open("r", encoding="utf-8") as f_in:
        examples = [json.loads(line) for line in f_in if line.strip()]

    flat_examples = []
    for ex in examples:
        segments = ex.get("segments", [])
        question = extract_question(segments)
        answer = segments_to_answer(segments)

        if not question.strip() or not answer.strip():
            continue

        flat_examples.append({
            "question": question,
            "answer": answer,
            "system": "",
            "sample_type": ex.get("sample_type", ""),
            "gold_answer": ex.get("gold_answer", ""),
        })

    with flat_path.open("w", encoding="utf-8") as f_out:
        for ex in flat_examples:
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  Converted {len(flat_examples)} examples → {flat_path}")


def step5_convert_flat(force: bool) -> None:
    """Convert protocol data to RelayLLM-compatible flat format."""
    print(f"\n{'='*60}")
    print("  Converting to RelayLLM-compatible format (question/answer/system)")
    print(f"{'='*60}")

    convert_to_flat_format(DEFAULT_TRAIN_PROTOCOL, DEFAULT_TRAIN_FLAT, force)
    convert_to_flat_format(DEFAULT_VAL_PROTOCOL, DEFAULT_VAL_FLAT, force)


# ---------------------------------------------------------------------------
# Summary & Upload
# ---------------------------------------------------------------------------

def print_summary() -> None:
    """Print dataset summary statistics."""
    print(f"\n{'='*60}")
    print("  Dataset Summary")
    print(f"{'='*60}")

    for label, path in [
        ("Raw Source Pool", RAW_DIR / "numinamath_cot_synthetic_math_seed_v0_1000.jsonl"),
        ("Raw Train", DEFAULT_TRAIN_RAW),
        ("Raw Validation", DEFAULT_VAL_RAW),
        ("Protocol Train", DEFAULT_TRAIN_PROTOCOL),
        ("Protocol Validation", DEFAULT_VAL_PROTOCOL),
        ("Flat Train (HF)", DEFAULT_TRAIN_FLAT),
        ("Flat Validation (HF)", DEFAULT_VAL_FLAT),
    ]:
        if path.exists():
            count = sum(1 for _ in path.open())
            size_kb = path.stat().st_size / 1024
            print(f"  {label:25s}: {count:>6d} examples  ({size_kb:.1f} KB)")
        else:
            print(f"  {label:25s}: NOT BUILT")

    # Sample type distribution for flat data
    for label, path in [
        ("Flat Train", DEFAULT_TRAIN_FLAT),
        ("Flat Validation", DEFAULT_VAL_FLAT),
    ]:
        if path.exists():
            with path.open() as f:
                examples = [json.loads(line) for line in f if line.strip()]
            types = Counter(e.get("sample_type", "unknown") for e in examples)
            print(f"\n  {label} sample types:")
            for t, c in sorted(types.items()):
                print(f"    {t:35s}: {c:>5d}  ({c/len(examples)*100:.1f}%)")

    # Decontamination status
    manifest_path = DEFAULT_SPLIT_MANIFEST
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        d = manifest.get("decontamination", {})
        print(f"\n  Decontamination:")
        print(f"    Exact match removed:     {d.get('exact_match_removed', '?')}")
        print(f"    Near-duplicate removed:  {d.get('near_duplicate_removed', '?')}")
        print(f"    Internal dup removed:    {d.get('internal_normalized_duplicate_removed', '?')}")
        print(f"    Held-out benchmarks:     {list(manifest.get('heldout_question_counts', {}).keys())}")

    # Show example in flat format
    if DEFAULT_TRAIN_FLAT.exists():
        print(f"\n{'='*60}")
        print("  Example (flat format, same as RelayLLM)")
        print(f"{'='*60}")
        with DEFAULT_TRAIN_FLAT.open() as f:
            for line in f:
                ex = json.loads(line)
                if ex.get("sample_type") == "ask_help":
                    print(f"\n  sample_type: {ex['sample_type']}")
                    print(f"  gold_answer: {ex['gold_answer']}")
                    print(f"  system:      '{ex['system']}'")
                    print(f"  question:    {ex['question'][:120]}...")
                    print(f"  answer:      {ex['answer'][:200]}...")
                    break


def upload_to_huggingface(hf_repo_id: str, private: bool) -> None:
    """Upload flat-format dataset to HuggingFace Hub."""
    try:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi
    except ImportError:
        print("\nERROR: Upload requires 'datasets' and 'huggingface_hub' packages.")
        print("       pip install datasets huggingface_hub")
        sys.exit(1)

    # Check login
    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"\nLogged in as: {user_info.get('name', user_info.get('fullname', '?'))}")
    except Exception:
        print("\nERROR: Not logged in to HuggingFace.")
        print("       Run: huggingface-cli login")
        sys.exit(1)

    if not DEFAULT_TRAIN_FLAT.exists() or not DEFAULT_VAL_FLAT.exists():
        print("ERROR: Flat-format data not found. Run with --build_only first.")
        sys.exit(1)

    def load_jsonl(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_examples = load_jsonl(DEFAULT_TRAIN_FLAT)
    val_examples = load_jsonl(DEFAULT_VAL_FLAT)

    # Build columnar format for HuggingFace
    def to_columns(examples: list[dict]) -> dict[str, list]:
        columns: dict[str, list] = {
            "question": [],
            "answer": [],
            "system": [],
            "sample_type": [],
            "gold_answer": [],
        }
        for ex in examples:
            columns["question"].append(ex["question"])
            columns["answer"].append(ex["answer"])
            columns["system"].append(ex.get("system", ""))
            columns["sample_type"].append(ex.get("sample_type", ""))
            columns["gold_answer"].append(ex.get("gold_answer", ""))
        return columns

    train_dataset = Dataset.from_dict(to_columns(train_examples))
    val_dataset = Dataset.from_dict(to_columns(val_examples))

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })

    print(f"\n  Uploading to: {hf_repo_id}")
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Validation examples: {len(val_dataset)}")
    print(f"  Private: {private}")
    print(f"  Format: question / answer / system (RelayLLM-compatible)")

    # Create dataset card
    card_content = f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - math
  - reasoning
  - collaborative-decoding
  - help-seeking-policy
  - sft
size_categories:
  - 1K<n<10K
---

# HSP Protocol SFT Dataset

Structured SFT training data for the **Help-Seeking Policy (HSP)** protocol.

## Format

This dataset follows the same format as RelayLLM SFT datasets for compatibility:

| Column | Type | Description |
|:-------|:-----|:------------|
| `question` | str | Math problem with prompt suffix |
| `answer` | str | Complete response (student text + teacher context interleaved) |
| `system` | str | System prompt (empty) |
| `sample_type` | str | Protocol sample type (see below) |
| `gold_answer` | str | Ground truth answer |

## HSP Protocol Tokens

The `answer` field contains three special action tokens with token budget control:

| Token | Semantics |
|:------|:----------|
| `<ASK>N</ASK>` | Student requests a reasoning hint (N = max teacher tokens) |
| `<VERIFY>N</VERIFY>` | Student requests verification (N = max teacher tokens) |
| `<ACCEPT>` | Student explicitly adopts the teacher's correction |

The integer N inside `<ASK>N</ASK>` and `<VERIFY>N</VERIFY>` controls how many
tokens the teacher model is allowed to generate, analogous to RelayLLM's `<call>N</call>`.

Teacher responses are wrapped in `<TEACHER_HELP>...</TEACHER_HELP>` (for ASK)
or `<TEACHER_REVIEW>...</TEACHER_REVIEW>` (for VERIFY).

## Sample Types

| Type | Description |
|:-----|:------------|
| `normal` | Independent reasoning, no interaction |
| `ask_help` | Student asks for help, receives teacher assistance |
| `verify_confirm` | Student correct, teacher confirms |
| `verify_accept_correction` | Student wrong, teacher corrects, student accepts |
| `verify_reject_bad_feedback` | Student correct, teacher wrong, student rejects |
| `verify_uncertain` | Teacher uncertain, student proceeds alone |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{hf_repo_id}")

# Same usage as RelayLLM SFT data
example = ds["train"][0]
print(example["question"])
print(example["answer"])

# Filter by sample type
ask_examples = ds["train"].filter(lambda x: x["sample_type"] == "ask_help")
```

## Source

Problems from [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
(category: `synthetic_math`). Decontaminated against MATH-500, GSM8K, AMC23,
Minerva, OlympiadBench, AIME 2024, AIME 2025.

## License

Apache 2.0 (inherited from the source dataset).
"""

    dataset_dict.push_to_hub(hf_repo_id, private=private)

    # Upload README
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=hf_repo_id,
        repo_type="dataset",
    )

    print(f"\n  Upload complete!")
    print(f"  URL: https://huggingface.co/datasets/{hf_repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HSP SFT datasets and optionally upload to HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build_only", action="store_true",
                       help="Only build datasets locally; do not upload.")
    mode.add_argument("--upload_only", action="store_true",
                       help="Only upload existing data; skip building.")

    parser.add_argument("--hf_repo_id", type=str, default=None,
                        help="HuggingFace repo ID, e.g. 'username/hsp-protocol-sft'.")
    parser.add_argument("--private", action="store_true",
                        help="Create private HuggingFace dataset.")
    parser.add_argument("--data_dir", type=str, default=str(default_generated_data_dir()),
                        help="Generated data root. Defaults to $INFOBUY_GENERATED_DATA.")

    parser.add_argument("--max_source_records", type=int, default=1000,
                        help="Number of source problems to fetch (default: 1000).")
    parser.add_argument("--train_size", type=int, default=800,
                        help="Number of training problems (default: 800).")
    parser.add_argument("--val_size", type=int, default=200,
                        help="Number of validation problems (default: 200).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if data already exists.")

    # Optional difficulty calibration (Step 2b) -- requires a GPU + reference model.
    parser.add_argument("--calibrate", action="store_true",
                        help="Annotate/select problems by difficulty before building protocol "
                             "data (needs GPU; runs the reference model).")
    parser.add_argument("--calibration_mode", choices=["tag", "stratify", "target", "band"], default="tag",
                        help="tag=keep all easy+hard, just annotate (SFT default); "
                             "stratify=balance easy/hard; target=narrow to --target_accuracy (RL).")
    parser.add_argument("--calibration_model", type=str, default="Qwen/Qwen3-8B",
                        help="Reference model for difficulty calibration (default: Qwen/Qwen3-8B).")
    parser.add_argument("--calibration_samples_k", type=int, default=4,
                        help="Samples per problem during calibration (default: 4).")
    parser.add_argument("--target_accuracy", type=float, default=0.70,
                        help="Target mean solve rate when --calibration_mode target (default: 0.70).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_data_dir(args.data_dir)

    if not args.build_only and not args.upload_only and args.hf_repo_id is None:
        print("ERROR: --hf_repo_id is required unless using --build_only.")
        print("       Example: python build_and_upload_hsp_dataset.py --build_only")
        print("       Example: python build_and_upload_hsp_dataset.py --hf_repo_id user/hsp-protocol-sft")
        sys.exit(1)

    if args.upload_only and args.hf_repo_id is None:
        print("ERROR: --hf_repo_id is required with --upload_only.")
        sys.exit(1)

    # ---- BUILD ----
    if not args.upload_only:
        print("\n" + "=" * 60)
        print("  HSP Dataset Builder")
        print("=" * 60)
        print(f"  Source records:  {args.max_source_records}")
        print(f"  Train size:     {args.train_size}")
        print(f"  Val size:       {args.val_size}")
        print(f"  Data dir:       {DATA_DIR}")
        print(f"  Seed:           {args.seed}")
        print(f"  Force rebuild:  {args.force}")
        print(f"  Calibrate:      {args.calibrate}"
              + (f" (mode={args.calibration_mode}, model={args.calibration_model})" if args.calibrate else ""))

        step1_fetch_source(args.max_source_records, args.force)
        step2_split_and_decontaminate(
            args.max_source_records, args.train_size, args.val_size, args.seed, args.force,
        )
        train_override = val_override = None
        if args.calibrate:
            train_override, val_override = step2b_calibrate_difficulty(
                args.train_size, args.val_size, args.calibration_model,
                args.calibration_samples_k, args.calibration_mode, args.target_accuracy, args.force,
            )
        step3_build_protocol_sft(
            args.train_size, args.val_size, args.seed, args.force,
            train_raw_override=train_override, val_raw_override=val_override,
        )
        step4_validate()
        step5_convert_flat(args.force)

    # ---- SUMMARY ----
    print_summary()

    # ---- UPLOAD ----
    if not args.build_only and args.hf_repo_id:
        upload_to_huggingface(args.hf_repo_id, args.private)
    elif args.build_only:
        print(f"\n{'='*60}")
        print("  Build complete! Review the data above.")
        print(f"{'='*60}")
        print(f"\n  To upload later, run:")
        print(f"  python build_and_upload_hsp_dataset.py --upload_only \\")
        print(f"      --hf_repo_id YOUR_USERNAME/hsp-protocol-sft")


if __name__ == "__main__":
    main()
