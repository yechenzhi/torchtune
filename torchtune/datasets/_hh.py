# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.config._errors import InstantiationError
from torchtune.config._utils import _get_component_from_path

from torchtune.data import PromptTemplate
from torchtune.datasets._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules import Tokenizer


class HhDataset(Dataset):
    """
    Class that supports any custom dataset with instruction-based prompts and a
    configurable template.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> format into template -> tokenize

    If the column/key names differ from the expected names in the `PromptTemplate`,
    then the `column_map` argument can be used to provide this mapping.

    Masking of the prompt during training is controlled by the `train_on_input` flag, which is
    set to `False` by default.
    - If `train_on_input` is True, the prompt is used during training and
    contributes to the loss.
    - If `train_on_input` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by HuggingFace's `load_dataset`
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        template (PromptTemplate): template used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use `column_map` to map them.
        transform (Optional[Callable]): transform to apply to the sample before formatting to the template.
            Default is None.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        transform: Optional[Callable] = None,
        column_map: Optional[Dict[str, str]] = None,
        train_on_input: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._transform = transform
        self._column_map = column_map
        self.train_on_input = train_on_input

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        transformed_sample = self._transform(sample) if self._transform else sample

        def extract_anthropic_prompt(prompt_and_response):
            """Extract the anthropic prompt from a prompt and response pair."""
            search_term = '\n\nAssistant:'
            search_term_idx = prompt_and_response.rfind(search_term)
            assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
            return prompt_and_response[:search_term_idx + len(search_term)]

        def split_prompt_and_responses(ex):
            prompt = extract_anthropic_prompt(ex['chosen'])
            chosen_response = ex['chosen'][len(prompt):]
            rejected_response = ex['rejected'][len(prompt):]
            return prompt, chosen_response, rejected_response
    
        prompt, chosen_response, _ = split_prompt_and_responses(transformed_sample)

        prompt_with_response = prompt + chosen_response

        encoded_prompt = self._tokenizer.encode(
            text=prompt, add_bos=True, add_eos=False
        )
        encoded_prompt_with_response = self._tokenizer.encode(
            text=prompt_with_response, add_bos=True, add_eos=True
        )
        labels = copy.deepcopy(encoded_prompt_with_response)

        if not self.train_on_input:
            labels[: len(encoded_prompt)] = [CROSS_ENTROPY_IGNORE_IDX] * len(
                encoded_prompt
            )

        assert len(encoded_prompt_with_response) == len(labels)

        return encoded_prompt_with_response, labels


def hh_dataset(
    tokenizer: Tokenizer,
    train_on_input: bool = False,
) -> HhDataset:
    """
    Build a configurable dataset with instruction prompts. This method should be
    used to configure a custom instruct dataset from the yaml config instead of
    using `InstructDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by HuggingFace's `load_dataset`
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        template (str): class name of template used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use `column_map` to map them.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.

    Returns:
        InstructDataset: the configured InstructDataset
    """
    return HhDataset(
        tokenizer=tokenizer,
        source='Anthropic/hh-rlhf',
        train_on_input=train_on_input,
        split="train",
    )
