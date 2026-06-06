# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str
    full_response: str
    full_response_length: int
    interaction_policy: str
    hsp_events: list[dict]
    teacher_tokens_used: int
    student_output_for_grading: str
    ask_count: int
    verify_count: int
    accept_count: int
    invalid_accept_count: int
    invalid_protocol_count: int
    denied_action_count: int


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_mask = data.batch["response_mask"].bool()
        valid_response_mask = data.batch.get("valid_response_mask", data.batch["response_mask"]).bool()
        for i in range(len(data)):
            cur_response_length = int(response_mask[i].sum().item())
            valid_response_ids = response_ids[i][response_mask[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            full_response_ids = response_ids[i][valid_response_mask[i]]
            stored_student_response = data.non_tensor_batch.get("student_output_for_grading", [None] * len(data))[i]
            if stored_student_response is not None:
                response_str = stored_student_response
            full_response = data.non_tensor_batch.get("full_transcript", [None] * len(data))[i]
            if full_response is None:
                full_response = self.tokenizer.decode(
                    full_response_ids, skip_special_tokens=self.config.skip_special_tokens
                )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "student_output_for_grading": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "full_response": full_response,
                    "full_response_length": int(valid_response_mask[i].sum().item()),
                    "interaction_policy": data.non_tensor_batch.get("interaction_policy", ["relay_call"] * len(data))[i],
                    "hsp_events": data.non_tensor_batch.get("hsp_events", [[]] * len(data))[i],
                    "teacher_tokens_used": int(data.non_tensor_batch.get("teacher_tokens_used", [0] * len(data))[i]),
                    "ask_count": int(data.non_tensor_batch.get("ask_count", [0] * len(data))[i]),
                    "verify_count": int(data.non_tensor_batch.get("verify_count", [0] * len(data))[i]),
                    "accept_count": int(data.non_tensor_batch.get("accept_count", [0] * len(data))[i]),
                    "invalid_accept_count": int(data.non_tensor_batch.get("invalid_accept_count", [0] * len(data))[i]),
                    "invalid_protocol_count": int(data.non_tensor_batch.get("invalid_protocol_count", [0] * len(data))[i]),
                    "denied_action_count": int(data.non_tensor_batch.get("denied_action_count", [0] * len(data))[i]),
                }
            )
            policy_positions = torch.nonzero(response_mask[i], as_tuple=False).flatten()
            if len(policy_positions):
                reward_tensor[i, policy_positions[-1]] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        prompts = data.batch["prompts"]
        # print(data.batch.keys())
        # print(data.non_tensor_batch.keys())
        # exit()
        response_mask = data.batch["response_mask"].bool()
        valid_response_mask = data.batch.get("valid_response_mask", data.batch["response_mask"]).bool()
        for i in range(len(data)):
            cur_response_length = int(response_mask[i].sum().item())
            valid_response_ids = response_ids[i][response_mask[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            full_response_ids = response_ids[i][valid_response_mask[i]]
            stored_student_response = data.non_tensor_batch.get("student_output_for_grading", [None] * len(data))[i]
            if stored_student_response is not None:
                response_str = stored_student_response
            full_response = data.non_tensor_batch.get("full_transcript", [None] * len(data))[i]
            if full_response is None:
                full_response = self.tokenizer.decode(
                    full_response_ids, skip_special_tokens=self.config.skip_special_tokens
                )
            prompt_str = self.tokenizer.decode(prompts[i], skip_special_tokens=self.config.skip_special_tokens)
            reward_inputs.append(
                {
                    "prompt": prompt_str,
                    "response": response_str,
                    "student_output_for_grading": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "full_response": full_response,
                    "full_response_length": int(valid_response_mask[i].sum().item()),
                    "interaction_policy": data.non_tensor_batch.get("interaction_policy", ["relay_call"] * len(data))[i],
                    "hsp_events": data.non_tensor_batch.get("hsp_events", [[]] * len(data))[i],
                    "teacher_tokens_used": int(data.non_tensor_batch.get("teacher_tokens_used", [0] * len(data))[i]),
                    "ask_count": int(data.non_tensor_batch.get("ask_count", [0] * len(data))[i]),
                    "verify_count": int(data.non_tensor_batch.get("verify_count", [0] * len(data))[i]),
                    "accept_count": int(data.non_tensor_batch.get("accept_count", [0] * len(data))[i]),
                    "invalid_accept_count": int(data.non_tensor_batch.get("invalid_accept_count", [0] * len(data))[i]),
                    "invalid_protocol_count": int(data.non_tensor_batch.get("invalid_protocol_count", [0] * len(data))[i]),
                    "denied_action_count": int(data.non_tensor_batch.get("denied_action_count", [0] * len(data))[i]),
                }
            )

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            policy_positions = torch.nonzero(response_mask[i], as_tuple=False).flatten()
            if len(policy_positions):
                reward_tensor[i, policy_positions[-1]] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
