"""Train a student model on structured HSP transcripts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

try:
    from .hsp_collator import CONTEXT_MARKER_TOKENS, HSPDataCollator, POLICY_ACTION_TOKENS, IGNORE_INDEX
    from .preflight_hsp import validate_dataset
except ImportError:
    from hsp_collator import CONTEXT_MARKER_TOKENS, HSPDataCollator, POLICY_ACTION_TOKENS, IGNORE_INDEX
    from preflight_hsp import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSP SFT training with student-only loss.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", required=True, help="Structured local JSONL/JSON path or Hugging Face dataset ID.")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--output_dir", default="./results_hsp_0.6B")
    parser.add_argument("--max_seq_length", type=int, default=12288)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit examples for smoke tests.")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1, help="Override training steps for smoke tests.")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", default="epoch")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_context_tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_structured_dataset(dataset_path: str, split: str):
    local_path = Path(dataset_path)
    if local_path.exists():
        extension = "json" if local_path.suffix.lower() in {".json", ".jsonl"} else local_path.suffix.lstrip(".")
        return load_dataset(extension, data_files=str(local_path), split="train")
    return load_dataset(dataset_path, split=split)


def main() -> None:
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dataset = load_structured_dataset(args.dataset, args.dataset_split)
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max_train_samples must be positive when supplied.")
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    protocol_report = validate_dataset(dataset)
    if protocol_report["errors"]:
        raise ValueError("HSP dataset failed protocol validation: " + "; ".join(protocol_report["errors"]))
    for warning in protocol_report["warnings"]:
        print(f"WARNING: {warning}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("HSP training requires a fast tokenizer for exact segment-level loss masks.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    added_tokens = list(POLICY_ACTION_TOKENS)
    if args.add_context_tokens:
        added_tokens.extend(CONTEXT_MARKER_TOKENS)
    added_count = tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})

    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if added_count:
        model.resize_token_embeddings(len(tokenizer))

    collator = HSPDataCollator(tokenizer=tokenizer, max_length=args.max_seq_length)

    def remains_trainable(example) -> bool:
        encoded = collator.encode_example(example)
        return encoded["interaction_complete"] and any(label != IGNORE_INDEX for label in encoded["labels"])

    original_size = len(dataset)
    dataset = dataset.filter(
        remains_trainable,
        desc="Removing incomplete or non-trainable examples after truncation",
    )
    if len(dataset) == 0:
        raise ValueError("No trainable HSP examples remain after tokenization/truncation.")

    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset} ({len(dataset)}/{original_size} trainable examples)")
    print(f"Policy action tokens: {POLICY_ACTION_TOKENS}")
    print(f"Additional context marker tokens enabled: {args.add_context_tokens}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        save_strategy=args.save_strategy,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    contract = {
        "max_seq_length": args.max_seq_length,
        "policy_action_tokens": list(POLICY_ACTION_TOKENS),
        "context_marker_tokens": list(CONTEXT_MARKER_TOKENS),
        "add_context_tokens": args.add_context_tokens,
    }
    contract_path = Path(args.output_dir) / "hsp_training_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
