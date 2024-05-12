# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import ChatbotArenaTemplate, InstructTemplate, Message

from torchtune.modules.tokenizers import Tokenizer


class ChatbotArenaDataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        template: InstructTemplate,
        transform: Optional[Callable] = None,
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset("csv", data_files=source, **load_dataset_kwargs)
        self.template = template
        self._transform = transform
        self.max_seq_len = max_seq_len
        self._data = self._data.filter(lambda x: x["winner_tie"] == 0)

    def __len__(self):
        return len(self._data)

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self._data))

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        for _ in range(10):
            sample = self._data[index]
            if (
                type(sample["response_a"]) != str
                or type(sample["response_b"]) != str
                or type(sample["prompt"]) != str
                or type(ast.literal_eval(sample["prompt"])[0]) != str
                or type(ast.literal_eval(sample["response_a"])[0]) != str
                or type(ast.literal_eval(sample["response_b"])[0]) != str
            ):

                index = self._rand_another()
                print("find another idx:", index)
                continue
            return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        transformed_sample = self._transform(sample) if self._transform else sample

        prompt_list = ast.literal_eval(transformed_sample["prompt"])
        response_a_list = ast.literal_eval(transformed_sample["response_a"])
        response_b_list = ast.literal_eval(transformed_sample["response_b"])

        conversation_a, conversation_b = [], []

        for prompt, response_a, response_b in zip(
            prompt_list, response_a_list, response_b_list
        ):
            prompt = self.template.format(prompt)
            conversation_a.append(Message(role="user", content=prompt, masked=True))
            conversation_a.append(Message(role="assistant", content=response_a))
            conversation_b.append(Message(role="user", content=prompt, masked=True))
            conversation_b.append(Message(role="assistant", content=response_b))

        conversation_a_input_ids, a_masks = self._tokenizer.tokenize_messages(
            conversation_a, self.max_seq_len
        )
        # conversation_a_labels = list(
        #     np.where(a_masks, CROSS_ENTROPY_IGNORE_IDX, conversation_a_input_ids)
        # )

        conversation_b_input_ids, b_masks = self._tokenizer.tokenize_messages(
            conversation_b, self.max_seq_len
        )
        # conversation_b_labels = list(
        #     np.where(b_masks, CROSS_ENTROPY_IGNORE_IDX, conversation_b_input_ids)
        # )

        if "winner_model_b" in sample and sample["winner_model_b"] == 1:
            conversation_a_input_ids, conversation_b_input_ids = (
                conversation_b_input_ids,
                conversation_a_input_ids,
            )
            a_masks, b_masks = b_masks, a_masks

        batch = dict(
            conversation_a_input_ids=conversation_a_input_ids,
            a_masks=a_masks,
            conversation_b_input_ids=conversation_b_input_ids,
            b_masks=b_masks,
        )

        return batch


def chatbot_arena_dataset(
    tokenizer: Tokenizer,
    source: str = "/root/autodl-tmp/kaggle/lmsys-chatbot-arena/train.csv",
    max_seq_len: int = 4096,
) -> ChatbotArenaDataset:
    return ChatbotArenaDataset(
        tokenizer=tokenizer,
        source=source,
        template=ChatbotArenaTemplate(),
        max_seq_len=max_seq_len,
        split="train",
    )
