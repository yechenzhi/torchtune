# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from datasets import load_dataset

from torch.utils.data import Dataset

from torchtune.data import InstructTemplate, Message

from torchtune.modules.tokenizers import Tokenizer

class ChatbotArenaDataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset("csv", data_files=source, **load_dataset_kwargs)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        def process(s):
            s=s.replace('null','"NULL"')
            s_list = s[1:-1].split('","')
            s_list[0] = s_list[0][1:]
            s_list[-1] = s_list[-1][:-1]
            return s_list

        prompt_list = process(sample["prompt"])
        response_a_list = process(sample["response_a"])
        response_b_list = process(sample["response_b"])

        if len(prompt_list) != len(response_a_list) or len(prompt_list) != len(
            response_b_list
        ):
            min_len = min(len(prompt_list), len(response_a_list), len(response_b_list))
            prompt_list = prompt_list[:min_len]
            response_a_list = response_a_list[:min_len]
            response_b_list = response_b_list[:min_len]

        assert len(prompt_list) == len(response_a_list) and len(prompt_list) == len(response_b_list)

        messages = []
        for prompt, response_a, response_b in zip(
                    prompt_list, response_a_list, response_b_list
                ):
            messages.append(Message(role="user", content=prompt)),
            messages.append(Message(role="assistant A", content=response_a)),
            messages.append(Message(role="assistant B", content=response_b)),

        # messages = self.chat_format.format(messages)
        tokens, _ = self._tokenizer.tokenize_messages(
            messages, max_seq_len=self.max_seq_len
        )

        label = 2
        if sample['winner_model_a'] == 1:
            label = 0
        elif sample['winner_model_b'] == 1:
            label = 1 
        else:
            label = 2
        
        data = dict(tokens=tokens, label=label)
        return data

    
def chatbot_arena_dataset(
    tokenizer: Tokenizer,
    source: str = "/root/dataDisk/lmsys/full_train.csv",
    max_seq_len: int = 4096,
) -> ChatbotArenaDataset:
    return ChatbotArenaDataset(
        tokenizer=tokenizer,
        source=source,
        max_seq_len=max_seq_len,
        split="train",
    )