"""Experiment I/O for feature-space adaptation and a small CPU sanity task."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def make_model(cfg):
    return FeatureMLP(**cfg["model"])


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dataset(cfg, path=None, split="train"):
    """NPZ keys x_train/y_train/x_val/y_val; x is float [N,D], y int [N]."""
    if path:
        with np.load(path, allow_pickle=False) as arrays:
            x = torch.tensor(arrays[f"x_{split}"], dtype=torch.float32)
            y = torch.tensor(arrays[f"y_{split}"], dtype=torch.long)
    else:
        # Fixed held-out task independent of the experiment's optimization seed.
        gen = torch.Generator().manual_seed(1701 if split == "train" else 1702)
        x = torch.randn(256 if split == "train" else 128, cfg["model"]["input_dim"], generator=gen)
        teacher_gen = torch.Generator().manual_seed(19)
        weight = torch.randn(cfg["model"]["input_dim"], cfg["model"]["num_classes"], generator=teacher_gen)
        y = (x @ weight).argmax(-1)
    if x.ndim != 2 or x.shape[1] != cfg["model"]["input_dim"] or y.shape != x.shape[:1]:
        raise ValueError("invalid feature or label shape")
    if len(y) == 0 or not torch.isfinite(x).all() or y.min() < 0 or y.max() >= cfg["model"]["num_classes"]:
        raise ValueError("empty/nonfinite features or out-of-range labels")
    return TensorDataset(x, y)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss = correct = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        count += len(y)
    return {"loss": loss / count, "accuracy": correct / count, "samples": count}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
