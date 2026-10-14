from __future__ import annotations

import os
from itertools import chain
from typing import Any

import datasets
import torch
from torch.utils.data import DataLoader, DistributedSampler


def load_packed_dataset(cache_path: str, seq_len: int):
    if not os.path.isdir(cache_path):
        raise FileNotFoundError(f"No packed cache at {cache_path}")
    dataset = datasets.load_from_disk(cache_path)
    if len(dataset) == 0:
        raise ValueError(f"Packed cache {cache_path} is empty.")
    actual_len = len(dataset[0]["input_ids"])
    if actual_len != seq_len:
        raise ValueError(
            f"Packed cache {cache_path} has rows of {actual_len} tokens but --seq-len is "
            f"{seq_len}. Point --packed-cache-path at the matching cache or rebuild it."
        )
    if "assistant_mask" not in dataset.column_names:
        raise ValueError(
            f"Packed cache {cache_path} has no `assistant_mask` column, so anchors could "
            f"only be sampled anywhere in the row -- which trains the diffusion view on prompt "
            f"tokens. Rebuild with --rebuild-packed-cache, or pass "
            f"--allow-missing-assistant-mask to accept it."
        )
    return dataset


def _collate(batch, keys=("input_ids", "assistant_mask")):
    import numpy as np

    out = {}
    for key in keys:
        if key in batch[0]:
            stacked = np.stack([row[key] for row in batch])
            out[key] = torch.from_numpy(stacked).long()
    return out


def create_packed_dataloader(
    cache_path: str,
    seq_len: int,
    micro_batch_size: int = 1,
    num_workers: int = 8,
    distributed: bool = False,
    shuffle: bool = True,
    seed: int = 42,
    require_assistant_mask: bool = True,
):
    dataset = load_packed_dataset(cache_path, seq_len) if require_assistant_mask else \
        datasets.load_from_disk(cache_path)
    columns = [c for c in ("input_ids", "assistant_mask") if c in dataset.column_names]
    dataset = dataset.select_columns(columns).with_format("numpy")

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
    )
    return loader


# Cache construction (one-off; CPU)

def assistant_token_mask(input_ids: list[int], tokenizer: Any) -> list[int]:
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    think_close = tokenizer.convert_tokens_to_ids("</think>")
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    if len(assistant_ids) != 1:
        raise ValueError(f"`assistant` is not a single token: {assistant_ids}")
    assistant_id = assistant_ids[0]

    num_tokens = len(input_ids)
    mask = [0] * num_tokens
    pos = 0
    while pos < num_tokens:
        is_assistant_header = (
            input_ids[pos] == im_start
            and pos + 1 < num_tokens
            and input_ids[pos + 1] == assistant_id
        )
        if not is_assistant_header:
            pos += 1
            continue

        content_start = pos + 2
        if content_start < num_tokens and input_ids[content_start] not in (im_start, im_end):
            content_start += 1  # the newline after the role name

        # Skip a reasoning block if present
        think_close_at = None
        for scan in range(content_start, num_tokens):
            if input_ids[scan] == im_end:
                break
            if input_ids[scan] == think_close:
                think_close_at = scan
                break
        if think_close_at is not None:
            content_start = think_close_at + 1
            if content_start < num_tokens and input_ids[content_start] not in (im_start, im_end):
                content_start += 1

        cursor = content_start
        while cursor < num_tokens and input_ids[cursor] != im_end:
            mask[cursor] = 1
            cursor += 1
        if cursor < num_tokens:  # the terminating <|im_end|> itself
            mask[cursor] = 1
        pos = cursor + 1
    return mask


def build_packed_cache(
    dataset_path,
    tokenizer,
    save_path: str,
    seq_len: int = 2048,
    num_proc: int = 16,
    max_samples: int | None = None,
    group_batch_size: int = 4096,
    shuffle: bool = True,
    seed: int = 42,
):
    """`messages` rows -> tokenized + `assistant_mask` -> packed fixed-length rows."""
    if os.path.isdir(dataset_path):
        raw = datasets.load_from_disk(dataset_path)
    else:
        ext = os.path.splitext(dataset_path)[1].lower()
        fmt = "json" if ext in (".json", ".jsonl") else "parquet" if ext == ".parquet" else None
        raw = datasets.load_dataset(fmt or dataset_path,
                                    **({"data_files": dataset_path} if fmt else {}),
                                    split="train")
    if "messages" not in raw.column_names:
        raise ValueError(f"Expected a `messages` column, got {raw.column_names}")
    raw = raw.select_columns(["messages"])
    if shuffle:
        raw = raw.shuffle(seed=seed)
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    def _tokenize(example):
        ids = list(tokenizer.apply_chat_template(
            example["messages"], tokenize=True, return_dict=True
        )["input_ids"])
        return {"input_ids": ids, "assistant_mask": assistant_token_mask(ids, tokenizer)}

    tokenized = raw.map(_tokenize, remove_columns=raw.column_names, num_proc=num_proc, desc="Tokenizing with chat template")

    def _group(examples):
        flat_ids = list(chain.from_iterable(examples["input_ids"]))
        flat_mask = list(chain.from_iterable(examples["assistant_mask"]))
        if len(flat_ids) != len(flat_mask):
            raise ValueError("input_ids / assistant_mask length mismatch after flatten.")
        total = (len(flat_ids) // seq_len) * seq_len
        return {
            "input_ids": [flat_ids[i : i + seq_len] for i in range(0, total, seq_len)],
            "assistant_mask": [flat_mask[i : i + seq_len] for i in range(0, total, seq_len)],
        }

    # Packed per map-batch
    packed = tokenized.map(_group, batched=True, batch_size=group_batch_size, num_proc=num_proc, remove_columns=tokenized.column_names, desc=f"Packing into {seq_len}-token rows")
    packed.save_to_disk(save_path)
    return packed
