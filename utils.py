"""Small experiment helpers shared by the two root entry points."""
import random

import numpy as np
import torch
import yaml


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.total += float(value) * count
        self.count += count

    @property
    def avg(self):
        return self.total / max(1, self.count)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path, opts=None):
    with open(path, encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    opts = opts or []
    if len(opts) % 2:
        raise ValueError('--opts expects pairs: dotted.key value')
    for key, value in zip(opts[::2], opts[1::2]):
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f'Unknown configuration key: {key}')
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def cuda_device(value='cuda'):
    device = torch.device(value)
    if device.type != 'cuda':
        raise ValueError('LFMA training and model evaluation require a CUDA device')
    return device
