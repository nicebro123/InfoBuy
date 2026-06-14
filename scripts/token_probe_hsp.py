#!/usr/bin/env python3
"""Probe whether an HSP SFT checkpoint can emit policy action tokens.

This is intentionally a post-SFT gate, not an accuracy evaluation. It loads
structured protocol examples, cuts each transcript immediately before a target
policy action, and checks whether the trained model assigns that action token a
reasonable next-token rank. Short deterministic generations are reported as a
diagnostic, but the default pass/fail criterion is rank based.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


POLICY_ACTION_TOKENS = ("<ASK>", "<VERIFY>", "<ACCEPT>")


def default_dataset_path() -> str:
    generated = os.environ.get("INFOBUY_GENERATED_DATA")
    if generated:
        return str(Path(generated) / "protocol" / "hsp_protocol_train_pilot_v1.jsonl")
    return "datasets/infobuy/protocol/hsp_protocol_train_pilot_v1.jsonl"


def default_model_path() -> str:
    checkpoint_root = os.environ.get("INFOBUY_CKPT")
    if checkpoint_root:
        return str(Path(checkpoint_root) / "sft" / "qwen3-0.6b-hsp-sft")
    return "checkpoints/sft/qwen3-0.6b-hsp-sft"


def load_examples(path: str) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"HSP protocol dataset not found: {source}")
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with source.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, list):
        return data
    raise ValueError(f"{source} must be JSONL or a JSON list.")


def initial_prompt(tokenizer: Any, segments: list[dict[str, Any]]) -> tuple[str, int]:
    messages = []
    first_generated_index = 0
    for first_generated_index, segment in enumerate(segments):
        if segment.get("source") in {"system", "user"} and not segment.get("loss", False):
            messages.append({"role": segment["source"], "content": str(segment.get("text", ""))})
            continue
        break
    else:
        first_generated_index = len(segments)

    if not messages:
        return "", 0
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text, first_generated_index


def build_probe_prefix(
    tokenizer: Any,
    example: dict[str, Any],
    expected_token: str,
) -> dict[str, Any] | None:
    segments = example.get("segments")
    if not isinstance(segments, list) or not segments:
        return None
    rendered, first_generated_index = initial_prompt(tokenizer, segments)
    for segment in segments[first_generated_index:]:
        text = str(segment.get("text", ""))
        if (
            segment.get("source") == "student"
            and bool(segment.get("loss", False))
            and expected_token in text
        ):
            target_index = text.index(expected_token)
            return {
                "id": example.get("id"),
                "sample_type": example.get("sample_type"),
                "expected_token": expected_token,
                "prefix": rendered + text[:target_index],
                "target_suffix": text[target_index : target_index + 48],
            }
        rendered += text
    return None


def collect_probes(
    tokenizer: Any,
    examples: list[dict[str, Any]],
    actions: list[str],
    probes_per_action: int,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    by_action: dict[str, list[dict[str, Any]]] = {action: [] for action in actions}
    for example in examples:
        if all(len(items) >= probes_per_action for items in by_action.values()):
            break
        for action in actions:
            if len(by_action[action]) >= probes_per_action:
                continue
            probe = build_probe_prefix(tokenizer, example, action)
            if probe is None:
                continue
            prompt_tokens = tokenizer.encode(probe["prefix"], add_special_tokens=False)
            if len(prompt_tokens) > max_prompt_tokens:
                continue
            probe["prompt_tokens"] = len(prompt_tokens)
            by_action[action].append(probe)

    missing = [action for action, probes in by_action.items() if not probes]
    if missing:
        raise ValueError("Could not build probes for actions: " + ", ".join(missing))
    return [probe for action in actions for probe in by_action[action]]


def parse_dtype(torch_module: Any, dtype_name: str, device: str) -> Any:
    if dtype_name == "auto":
        return torch_module.bfloat16 if device == "cuda" else torch_module.float32
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[dtype_name]


def resolve_device(torch_module: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def validate_action_token_ids(tokenizer: Any, actions: list[str]) -> dict[str, int]:
    token_ids: dict[str, int] = {}
    errors = []
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for action in actions:
        ids = tokenizer.encode(action, add_special_tokens=False)
        if len(ids) != 1:
            errors.append(f"{action} is not a single tokenizer token: {ids}")
            continue
        if unk_token_id is not None and ids[0] == unk_token_id:
            errors.append(f"{action} maps to unk_token_id")
            continue
        token_ids[action] = ids[0]
    if errors:
        raise ValueError("; ".join(errors))
    return token_ids


def probe_model(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "token_probe_hsp requires torch and transformers. Install the training environment first."
        ) from error

    actions = [item.strip() for item in args.actions.split(",") if item.strip()]
    if not actions:
        raise ValueError("--actions must contain at least one token.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("HSP token probe requires a fast tokenizer.")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    action_token_ids = validate_action_token_ids(tokenizer, actions)

    examples = load_examples(args.dataset)
    probes = collect_probes(
        tokenizer=tokenizer,
        examples=examples,
        actions=actions,
        probes_per_action=args.probes_per_action,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    device = resolve_device(torch, args.device)
    dtype = parse_dtype(torch, args.dtype, device)
    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if device != "cpu":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.to(device)
    model.eval()

    results = []
    for probe in probes:
        encoded = tokenizer(probe["prefix"], return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        expected_id = action_token_ids[probe["expected_token"]]
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()[0]
            expected_logit = logits[expected_id]
            rank = int((logits > expected_logit).sum().item() + 1)
            probability = float(torch.softmax(logits, dim=-1)[expected_id].item())

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.temperature > 0:
                generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
            else:
                generation_kwargs["do_sample"] = False
            generated = model.generate(**encoded, **generation_kwargs)
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        results.append({
            "id": probe["id"],
            "sample_type": probe["sample_type"],
            "expected_token": probe["expected_token"],
            "expected_token_id": expected_id,
            "prompt_tokens": probe["prompt_tokens"],
            "rank": rank,
            "probability": probability,
            "rank_hit": rank <= args.max_rank,
            "generated_text": generated_text,
            "generation_hit": probe["expected_token"] in generated_text,
            "target_suffix": probe["target_suffix"],
        })

    by_action: dict[str, dict[str, Any]] = {}
    for action in actions:
        action_results = [result for result in results if result["expected_token"] == action]
        by_action[action] = {
            "count": len(action_results),
            "rank_hit_count": sum(1 for result in action_results if result["rank_hit"]),
            "generation_hit_count": sum(1 for result in action_results if result["generation_hit"]),
            "median_rank": sorted(result["rank"] for result in action_results)[len(action_results) // 2],
            "best_rank": min(result["rank"] for result in action_results),
            "worst_rank": max(result["rank"] for result in action_results),
        }

    rank_hit_rate = sum(1 for result in results if result["rank_hit"]) / len(results)
    generation_hit_rate = sum(1 for result in results if result["generation_hit"]) / len(results)
    ok = (
        rank_hit_rate >= args.min_rank_hit_rate
        and generation_hit_rate >= args.min_generation_hit_rate
        and all(stats["rank_hit_count"] > 0 for stats in by_action.values())
    )
    return {
        "ok": ok,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "device": device,
        "actions": actions,
        "max_rank": args.max_rank,
        "min_rank_hit_rate": args.min_rank_hit_rate,
        "min_generation_hit_rate": args.min_generation_hit_rate,
        "rank_hit_rate": rank_hit_rate,
        "generation_hit_rate": generation_hit_rate,
        "by_action": by_action,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe HSP policy action token readiness after SFT.")
    parser.add_argument("--model_path", default=default_model_path())
    parser.add_argument("--dataset", default=default_dataset_path())
    parser.add_argument("--actions", default=",".join(POLICY_ACTION_TOKENS))
    parser.add_argument("--probes_per_action", type=int, default=4)
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--max_rank", type=int, default=100)
    parser.add_argument("--min_rank_hit_rate", type=float, default=0.80)
    parser.add_argument("--min_generation_hit_rate", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--no_fail", action="store_true", help="Always exit 0 after writing the report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = probe_model(args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if args.as_json:
        print(rendered)
    else:
        print(f"HSP token probe: {'PASS' if report['ok'] else 'FAIL'}")
        print(f"Model: {report['model_path']}")
        print(f"Dataset: {report['dataset']}")
        print(f"Rank hit rate: {report['rank_hit_rate']:.3f}")
        print(f"Generation hit rate: {report['generation_hit_rate']:.3f}")
        for action, stats in report["by_action"].items():
            print(
                f"  {action}: rank_hits={stats['rank_hit_count']}/{stats['count']} "
                f"median_rank={stats['median_rank']} best={stats['best_rank']} worst={stats['worst_rank']} "
                f"generation_hits={stats['generation_hit_count']}/{stats['count']}"
            )
        if args.output:
            print(f"Report: {args.output}")
    if not report["ok"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
