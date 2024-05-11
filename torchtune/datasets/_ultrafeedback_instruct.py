# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data import StackExchangedPairedTemplate
from torchtune.datasets._instruct import InstructDataset
from torchtune.modules.tokenizers import Tokenizer


def ultrafeedback_instruct_dataset(
    tokenizer: Tokenizer,
    source: str = "argilla/ultrafeedback-binarized-preferences-cleaned",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
) -> InstructDataset:
    """
    Family of preference datasets similar to `StackExchangePaired data
    <https://huggingface.co/datasets/lvwerra/stack-exchange-paired>`_.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
            Default is 1024.

    Returns:
        PreferenceDataset: The preference dataset built from source paired data.
    """
    def transform(sample):
        transformed_sample = dict(
            prompt = sample['prompt'],
            output = sample['chosen'][1]['content'],
        )
        return transformed_sample 
    return InstructDataset(
        tokenizer=tokenizer,
        source=source,
        template=StackExchangedPairedTemplate(),
        column_map={
            "prompt": "prompt",
            "output": "output",
        },
        train_on_input=train_on_input,
        transform = transform,
        max_seq_len=max_seq_len,
        split="train",
    )
