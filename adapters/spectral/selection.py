from __future__ import annotations

import torch


def topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    """Stable magnitude order. ties resolve to the lower flattened index."""
    if not 1 <= k <= values.numel():
        raise ValueError(f"k must lie in [1, {values.numel()}], got {k}")
    return torch.argsort(values.flatten(), descending=True, stable=True)[:k]
