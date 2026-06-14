"""Qwen2-VL position-id helpers.

This module contains the mRoPE position-id calculation used by
``verl.utils.dataset.RLHFDataset`` for Qwen2-VL/Qwen2.5-VL processors.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import ProcessorMixin


def _token_id(processor: ProcessorMixin, token: str, fallback_attr: str) -> int:
    tokenizer = processor.tokenizer
    token_id = getattr(tokenizer, fallback_attr, None)
    if token_id is not None:
        return int(token_id)
    converted = tokenizer.convert_tokens_to_ids(token)
    if converted is None:
        raise ValueError(f"tokenizer cannot resolve {token!r}")
    return int(converted)


def _merge_size(processor: ProcessorMixin) -> int:
    image_processor = getattr(processor, "image_processor", None)
    return int(getattr(image_processor, "merge_size", 2))


def _spatial_positions(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    merge_size: int,
) -> torch.Tensor:
    llm_grid_h = grid_h // merge_size
    llm_grid_w = grid_w // merge_size
    text_len = grid_t * llm_grid_h * llm_grid_w
    t_index = torch.arange(grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
    h_index = (
        torch.arange(llm_grid_h)
        .view(1, -1, 1)
        .expand(grid_t, -1, llm_grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(llm_grid_w)
        .view(1, 1, -1)
        .expand(grid_t, llm_grid_h, -1)
        .flatten()
    )
    return torch.stack([t_index, h_index, w_index], dim=0).to(torch.long).reshape(3, text_len)


def get_rope_index(
    processor: ProcessorMixin,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor | list[float]] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return Qwen2-VL mRoPE position ids for one prompt.

    Args:
        processor: Qwen2-VL/Qwen2.5-VL processor.
        input_ids: One-dimensional token ids for one prompt.
        image_grid_thw: Image grids with rows ``[t, h, w]``.
        video_grid_thw: Video grids with rows ``[t, h, w]``.
        second_per_grid_ts: Accepted for API compatibility. The current dataset
            only needs discrete mRoPE indices.
        attention_mask: Optional one-dimensional attention mask.

    Returns:
        Tensor of shape ``(3, seq_len)``.
    """

    if input_ids.dim() != 1:
        raise ValueError("get_rope_index expects one-dimensional input_ids")

    del second_per_grid_ts
    device = input_ids.device
    seq_len = input_ids.size(0)
    if attention_mask is None:
        valid_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    else:
        valid_mask = attention_mask.to(device=device).bool()

    position_ids = torch.zeros(3, seq_len, dtype=torch.long, device=device)
    valid_input_ids = input_ids[valid_mask].detach().cpu().tolist()
    if not valid_input_ids:
        return position_ids

    image_token_id = _token_id(processor, "<|image_pad|>", "image_token_id")
    video_token_id = _token_id(processor, "<|video_pad|>", "video_token_id")
    vision_start_token_id = _token_id(processor, "<|vision_start|>", "vision_start_token_id")
    merge_size = _merge_size(processor)
    image_grid_thw = image_grid_thw.detach().cpu() if image_grid_thw is not None else torch.empty((0, 3), dtype=torch.long)
    video_grid_thw = video_grid_thw.detach().cpu() if video_grid_thw is not None else torch.empty((0, 3), dtype=torch.long)
    image_index = 0
    video_index = 0
    cursor = 0
    text_offset = 0
    chunks: list[torch.Tensor] = []

    while cursor < len(valid_input_ids):
        next_image = next_video = len(valid_input_ids)
        for index in range(cursor, len(valid_input_ids)):
            if valid_input_ids[index] == vision_start_token_id and index + 1 < len(valid_input_ids):
                next_token = valid_input_ids[index + 1]
                if next_token == image_token_id:
                    next_image = index + 1
                    break
                if next_token == video_token_id:
                    next_video = index + 1
                    break

        next_vision = min(next_image, next_video)
        if next_vision == len(valid_input_ids):
            text_len = len(valid_input_ids) - cursor
            if text_len:
                text_pos = torch.arange(text_len).view(1, -1).expand(3, -1) + text_offset
                chunks.append(text_pos)
            break

        text_len = next_vision - cursor
        if text_len:
            text_pos = torch.arange(text_len).view(1, -1).expand(3, -1) + text_offset
            chunks.append(text_pos)
            text_offset += text_len

        is_image = next_image <= next_video
        if is_image:
            if image_index >= len(image_grid_thw):
                raise ValueError("image token count exceeds image_grid_thw rows")
            grid_t, grid_h, grid_w = [int(value) for value in image_grid_thw[image_index]]
            image_index += 1
        else:
            if video_index >= len(video_grid_thw):
                raise ValueError("video token count exceeds video_grid_thw rows")
            grid_t, grid_h, grid_w = [int(value) for value in video_grid_thw[video_index]]
            video_index += 1

        spatial = _spatial_positions(grid_t, grid_h, grid_w, merge_size) + text_offset
        chunks.append(spatial)
        text_offset = int(spatial.max().item()) + 1
        cursor = next_vision + spatial.size(1)

    if chunks:
        valid_position_ids = torch.cat(chunks, dim=1).to(device=device)
    else:
        valid_position_ids = torch.empty((3, 0), dtype=torch.long, device=device)

    if valid_position_ids.size(1) < int(valid_mask.sum().item()):
        tail_len = int(valid_mask.sum().item()) - valid_position_ids.size(1)
        tail = torch.arange(tail_len, device=device).view(1, -1).expand(3, -1)
        tail = tail + (int(valid_position_ids.max().item()) + 1 if valid_position_ids.numel() else 0)
        valid_position_ids = torch.cat([valid_position_ids, tail], dim=1)
    elif valid_position_ids.size(1) > int(valid_mask.sum().item()):
        valid_position_ids = valid_position_ids[:, : int(valid_mask.sum().item())]

    position_ids[:, valid_mask] = valid_position_ids
    return position_ids
