# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset


PROMPT_TEMPLATE = (
    "You are given a title of a blog post below. Write the content of the post.\n\n"
    "### Title: {title}\n\n"
    "### Content:\n\n"
)


class NikPostDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split="train"):
        all_posts = json.load(open(dataset_config.data_path))

        # Train on data before 2025
        training_data = [
            post for post in all_posts
            if post['date'] < '2025'
        ]
        if len(training_data) == 0:
            raise Exception('No training data. Check input file.')

        # Sort by date if not already sorted (just in case)
        training_data.sort(key=lambda x: x["date"])

        val_len = int(len(training_data) * 0.05)
        if val_len == 0:
            raise Exception('Too few training data')

        step = len(training_data) // val_len

        val_indices = set(range(0, len(training_data), step))

        if split == "train":
            self.ann = [post for i, post in enumerate(
                training_data) if i not in val_indices]
        else:
            self.ann = [post for i, post in enumerate(
                training_data) if i in val_indices]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        prompt = PROMPT_TEMPLATE.format_map(ann)
        example = prompt + ann["content"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }
