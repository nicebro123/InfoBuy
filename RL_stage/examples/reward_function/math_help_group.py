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

import re
from typing import Any, List, Dict
from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    
    processed_data = []
    passed_prompts = set()

    for reward_input in reward_inputs:
        prompt = reward_input["prompt"]
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])

        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        format_score = format_reward(response)

        is_correct = accuracy_score > 0.5

        if_call = '<call>' in response
        call_num = sum(min(int(num), 4096) for num in re.findall(r"<call>\s*(\d+)\s*</call>", response))
        call_ratio = min(call_num / reward_input["response_length"], 1.0)
        item_data = {
            "prompt": prompt,
            "accuracy_score": accuracy_score,
            "format_score": format_score,
            "if_call": if_call,
            "call_num": call_num,
            "call_ratio": call_ratio,
            "is_correct": is_correct
        }
        processed_data.append(item_data)

        if is_correct:
            passed_prompts.add(prompt)

    # Build per-prompt-group flags for GRPO advantage normalization
    group_has_no_call_correct: Dict[str, bool] = {}
    group_has_call_correct: Dict[str, bool] = {}
    for item in processed_data:
        prompt = item["prompt"]
        if prompt not in group_has_no_call_correct:
            group_has_no_call_correct[prompt] = False
            group_has_call_correct[prompt] = False
        if item["is_correct"]:
            if item["if_call"]:
                group_has_call_correct[prompt] = True
            else:
                group_has_no_call_correct[prompt] = True

    scores = []
    for item in processed_data:
        prompt = item["prompt"]
        is_correct = item["is_correct"]
        if_call = item["if_call"]
        call_ratio = item["call_ratio"]
        has_no_call_correct = group_has_no_call_correct[prompt]
        has_call_correct = group_has_call_correct[prompt]

        overall_score = 0.0

        if has_no_call_correct:
            if is_correct and not if_call:
                overall_score = 1.5
            elif is_correct and if_call:
                overall_score = 1.0 - call_ratio
            else:
                overall_score = 0.0
        elif has_call_correct:
            if is_correct and if_call:
                overall_score = 1.0 - call_ratio
            elif not is_correct and not if_call:
                overall_score = -1.0
            elif not is_correct and if_call:
                overall_score = 0.0
        else:
            if if_call:
                overall_score = call_ratio
            else:
                overall_score = 0.0

        pass_at_n_score = 1.0 if prompt in passed_prompts else 0.0

        scores.append(
            {
                "overall": overall_score,
                "format": item["format_score"],
                "mean@32": item["accuracy_score"],
                "pass@32": pass_at_n_score,
                "call_ratio": item["call_ratio"],
                'if_call': item["if_call"],
                'call_num': item["call_num"],
            }
        )

    return scores