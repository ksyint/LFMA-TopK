"""Deterministic epoch sampling and balanced subsets for benchmark runs."""

import math
import hashlib
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler


def class_indices(labels):
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        value = int(label)
        if value < 0 or value != label:
            raise ValueError("Sampling requires nonnegative integer labels")
        groups[value].append(index)
    if not groups:
        raise ValueError("Sampling requires labeled examples")
    return dict(sorted(groups.items()))


def stratified_subset(labels, count, seed=42, minimum_per_class=1):
    groups = class_indices(labels)
    if not len(groups) * minimum_per_class <= count <= len(labels):
        raise ValueError("Subset size cannot satisfy the per-class minimum")
    if any(len(values) < minimum_per_class for values in groups.values()):
        raise ValueError("A class has fewer examples than the requested minimum")
    allocations = {key: minimum_per_class for key in groups}
    remaining = count - sum(allocations.values())
    while remaining:
        eligible = [key for key in groups if allocations[key] < len(groups[key])]
        selected = max(
            eligible,
            key=lambda key: (
                len(groups[key]) * count / len(labels) - allocations[key],
                -key,
            ),
        )
        allocations[selected] += 1
        remaining -= 1
    rng = np.random.default_rng(seed)
    result = []
    for key, values in groups.items():
        result.extend(rng.choice(values, allocations[key], replace=False).tolist())
    rng.shuffle(result)
    return result


class EpochSampler(Sampler):
    def __init__(self, size, seed=42, shuffle=True, rank=0, world_size=1):
        if size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid dataset size or distributed sampling coordinates")
        self.size = int(size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = int(epoch)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.size, generator=generator).tolist()
        else:
            indices = list(range(self.size))
        return iter(indices[self.rank :: self.world_size])

    def __len__(self):
        return max(0, math.ceil((self.size - self.rank) / self.world_size))

    def state_dict(self):
        return {
            "size": self.size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "rank": self.rank,
            "world_size": self.world_size,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in expected.keys() - {"epoch"}:
            if state[key] != expected[key]:
                raise ValueError(f"Sampler resume changed {key}")
        self.set_epoch(state["epoch"])


class LengthBucketSampler(Sampler):
    def __init__(self, lengths, batch_size, seed=42, bucket_batches=20):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        if self.lengths.ndim != 1 or len(self.lengths) == 0:
            raise ValueError("Token lengths must be a nonempty vector")
        if np.any(self.lengths < 1) or min(batch_size, bucket_batches) < 1:
            raise ValueError("Token and bucket sizes must be positive")
        self.batch_size = int(batch_size)
        self.bucket_size = int(batch_size * bucket_batches)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError('Epoch must be nonnegative')
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths))
        batches = []
        for offset in range(0, len(order), self.bucket_size):
            bucket = order[offset : offset + self.bucket_size]
            bucket = bucket[np.argsort(self.lengths[bucket], kind="stable")]
            batches.extend(
                bucket[index : index + self.batch_size].tolist()
                for index in range(0, len(bucket), self.batch_size)
            )
        tail = batches.pop() if len(batches[-1]) != self.batch_size else None
        rng.shuffle(batches)
        if tail is not None:
            batches.append(tail)
        return iter(index for batch in batches for index in batch)

    def state_dict(self):
        return {
            'format': 'lfma-length-sampler-v1',
            'lengths_sha256': hashlib.sha256(self.lengths.tobytes()).hexdigest(),
            'samples': len(self.lengths),
            'batch_size': self.batch_size,
            'bucket_size': self.bucket_size,
            'seed': self.seed,
            'epoch': self.epoch,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in expected.keys() - {'epoch'}:
            if state.get(key) != expected[key]:
                raise ValueError(f'Length sampler resume changed {key}')
        self.set_epoch(state['epoch'])


def token_lengths(dataset, tokenizer, task, max_length, chunk_size=256):
    from lfma.core import GLUE_TASKS

    if task not in GLUE_TASKS:
        raise ValueError('Length sampling requires a text classification or STS-B task')
    if tokenizer is None or chunk_size < 1:
        raise ValueError('Length sampling requires the task tokenizer and a positive chunk')
    first, second, _ = GLUE_TASKS[task]
    records = getattr(dataset, 'records', dataset)
    lengths = []
    for offset in range(0, len(records), chunk_size):
        rows = [records[index] for index in range(offset, min(len(records), offset + chunk_size))]
        batch = tokenizer(
            [row[first] for row in rows],
            [row[second] for row in rows] if second else None,
            truncation=True, padding=False, max_length=max_length,
            return_attention_mask=False, return_token_type_ids=False,
        )
        batch_lengths = [len(tokens) for tokens in batch['input_ids']]
        if len(batch_lengths) != len(rows) or any(
            length < 1 or length > max_length for length in batch_lengths
        ):
            raise ValueError('Tokenizer returned invalid lengths for the sampler')
        lengths.extend(batch_lengths)
    return lengths


def seed_worker(worker_id):
    import random

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def sampler_for(dataset, config, split, processor=None):
    options = config["data"].get("sampling", {})
    if split != "train":
        return EpochSampler(len(dataset), shuffle=False)
    seed = config["train"]["seed"]
    strategy = options.get("strategy", "shuffle")
    if strategy == "shuffle":
        return EpochSampler(len(dataset), seed=seed)
    if strategy == "length":
        lengths = token_lengths(dataset, processor, config['data']['task'],
                                config['data']['max_length'])
        return LengthBucketSampler(
            lengths,
            config["train"]["batch_size"],
            seed,
            options.get("bucket_batches", 20),
        )
    raise ValueError(f"Unknown training sampling strategy: {strategy}")
