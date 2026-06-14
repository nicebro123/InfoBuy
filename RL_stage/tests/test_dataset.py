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

import torch
from PIL import Image
from PIL.Image import Image as ImageObject

from verl.utils import dataset as dataset_module
from verl.utils.dataset import RLHFDataset


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return dict(self.rows[index])

    def filter(self, *args, **kwargs):
        return self


class FakeTokenizer:
    pad_token_id = 0
    image_token_id = 1001
    video_token_id = 1002
    vision_start_token_id = 1000

    def convert_tokens_to_ids(self, token):
        return {
            "<|vision_start|>": self.vision_start_token_id,
            "<|image_pad|>": self.image_token_id,
            "<|video_pad|>": self.video_token_id,
        }[token]

    def encode(self, text, add_special_tokens=False):
        return [11, 12, self.vision_start_token_id, self.image_token_id, 13, 14]


class Qwen2VLImageProcessor:
    merge_size = 1


class FakeProcessor:
    tokenizer = FakeTokenizer()
    image_processor = Qwen2VLImageProcessor()
    model_input_names = []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, enable_thinking=False):
        return "USER:<image> What is shown?\nASSISTANT:"

    def __call__(self, images, text, add_special_tokens=False, return_tensors=None):
        return {
            "input_ids": torch.tensor(
                [[11, 12, self.tokenizer.vision_start_token_id, self.tokenizer.image_token_id, 13, 14]]
            ),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


def test_image_dataset_uses_local_image_and_qwen2vl_position_ids(monkeypatch):
    image = Image.new("RGB", (4, 4), color="white")
    rows = [{"problem": "<image> What is shown?", "answer": "48", "images": [image]}]
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: FakeDataset(rows))

    dataset = RLHFDataset(
        data_path="local_vision@test",
        tokenizer=FakeTokenizer(),
        processor=FakeProcessor(),
        prompt_key="problem",
        answer_key="answer",
        image_key="images",
        max_prompt_length=6,
        truncation="right",
        filter_overlong_prompts=False,
    )

    item = dataset[0]
    expected_token_ids = torch.tensor([11, 12, 1000, 1001, 13, 14])
    assert set(item.keys()) == {
        "input_ids",
        "attention_mask",
        "position_ids",
        "raw_prompt_ids",
        "ground_truth",
        "multi_modal_data",
    }
    assert torch.all(item["input_ids"] == expected_token_ids)
    assert torch.all(item["attention_mask"] == torch.ones(6))
    assert torch.all(item["position_ids"] == torch.arange(6).unsqueeze(0).expand(3, -1))
    assert list(item["position_ids"].size()) == [3, 6]
    assert item["raw_prompt_ids"] == expected_token_ids.tolist()
    assert item["ground_truth"] == "48"
    assert isinstance(item["multi_modal_data"]["images"][0], ImageObject)
