from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import TensorDataset


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
