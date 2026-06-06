"""Segment-aware collator for HSP supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


IGNORE_INDEX = -100
POLICY_ACTION_TOKENS = ["<ASK>", "</ASK>", "<VERIFY>", "</VERIFY>", "<ACCEPT>"]
CONTEXT_MARKER_TOKENS = [
    "<TEACHER_HELP>",
    "</TEACHER_HELP>",
    "<TEACHER_REVIEW>",
    "</TEACHER_REVIEW>",
    "<ENVIRONMENT_NOTICE>",
    "</ENVIRONMENT_NOTICE>",
]


@dataclass
class HSPDataCollator:
    tokenizer: Any
    max_length: int = 12288
    append_eos_token: bool = True

    def _render_example(self, feature: dict[str, Any]) -> tuple[str, list[tuple[int, int]], bool]:
        segments = feature.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("Each HSP example must contain a non-empty segments list.")
        for segment in segments:
            if segment.get("loss", False) and segment.get("source") != "student":
                raise ValueError("Only source=student segments may have loss=true in HSP training data.")

        initial_messages = []
        first_generated_index = 0
        for first_generated_index, segment in enumerate(segments):
            if segment.get("source") in {"system", "user"} and not segment.get("loss", False):
                initial_messages.append({"role": segment["source"], "content": segment["text"]})
                continue
            break
        else:
            first_generated_index = len(segments)

        if initial_messages:
            text = self.tokenizer.apply_chat_template(
                initial_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = ""
            first_generated_index = 0

        loss_spans: list[tuple[int, int]] = []
        last_segment_has_loss = False
        for segment in segments[first_generated_index:]:
            segment_text = str(segment.get("text", ""))
            start = len(text)
            text += segment_text
            end = len(text)
            last_segment_has_loss = bool(segment.get("loss", False))
            if last_segment_has_loss and end > start:
                loss_spans.append((start, end))
        return text, loss_spans, last_segment_has_loss

    @staticmethod
    def _in_loss_span(start: int, end: int, loss_spans: list[tuple[int, int]]) -> bool:
        return end > start and any(start >= span_start and end <= span_end for span_start, span_end in loss_spans)

    def _encode_with_offsets(
        self, text: str, loss_spans: list[tuple[int, int]]
    ) -> tuple[list[int], list[int], bool]:
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            full_input_ids = list(encoded["input_ids"])
            full_offsets = list(encoded["offset_mapping"])
            truncated = len(full_input_ids) > self.max_length
            input_ids = full_input_ids[: self.max_length]
            offsets = full_offsets[: self.max_length]
            labels = [
                token_id if self._in_loss_span(start, end, loss_spans) else IGNORE_INDEX
                for token_id, (start, end) in zip(input_ids, offsets)
            ]
            return input_ids, labels, truncated
        except (NotImplementedError, TypeError, ValueError, KeyError):
            raise ValueError(
                "HSPDataCollator requires a fast tokenizer with return_offsets_mapping support "
                "so teacher spans can be masked without boundary ambiguity."
            )

    def encode_example(self, feature: dict[str, Any]) -> dict[str, Any]:
        text, loss_spans, last_segment_has_loss = self._render_example(feature)
        input_ids, labels, truncated = self._encode_with_offsets(text, loss_spans)
        trainable_text = "".join(text[start:end] for start, end in loss_spans)
        has_interaction_action = any(token in trainable_text for token in POLICY_ACTION_TOKENS)
        interaction_complete = not (truncated and has_interaction_action)
        eos_id = self.tokenizer.eos_token_id
        if (
            self.append_eos_token
            and eos_id is not None
            and len(input_ids) < self.max_length
            and (not input_ids or input_ids[-1] != eos_id)
        ):
            input_ids.append(eos_id)
            labels.append(eos_id if last_segment_has_loss else IGNORE_INDEX)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "interaction_complete": interaction_complete,
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        import torch

        encoded_features = [self.encode_example(feature) for feature in features]
        if any(not feature["interaction_complete"] for feature in encoded_features):
            raise ValueError("An HSP interaction was truncated before its observation/continuation was complete.")
        if any(not any(label != IGNORE_INDEX for label in feature["labels"]) for feature in encoded_features):
            raise ValueError("An HSP batch item has no trainable student tokens after truncation.")

        max_batch_length = max(len(feature["input_ids"]) for feature in encoded_features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        for feature in encoded_features:
            padding_length = max_batch_length - len(feature["input_ids"])
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                batch_input_ids.append([pad_id] * padding_length + feature["input_ids"])
                batch_labels.append([IGNORE_INDEX] * padding_length + feature["labels"])
                batch_attention_mask.append([0] * padding_length + [1] * len(feature["input_ids"]))
            else:
                batch_input_ids.append(feature["input_ids"] + [pad_id] * padding_length)
                batch_labels.append(feature["labels"] + [IGNORE_INDEX] * padding_length)
                batch_attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding_length)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }
