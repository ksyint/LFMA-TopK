from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from adapters.spectral import topk_indices


class FourierLinear(nn.Module):
    """W = W0 + alpha * Re(IFFT2(S(c))). exactly 2*k real adapter scalars.

    Shapes use PyTorch's [out_features, in_features] weight convention. The
    support is selected once from FFT2(delta_init), including its coefficients.
    No conjugate-symmetry constraint is imposed, as in the paper.
    """
    def __init__(self, base_layer: nn.Linear, delta_init: torch.Tensor,
                 top_k_ratio: float = 0.05, alpha: float = 12.0, k: int = None):
        super().__init__()
        if tuple(delta_init.shape) != tuple(base_layer.weight.shape):
            raise ValueError("delta_init must have the same [out, in] shape as the weight")
        if not 0 < top_k_ratio <= 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("require 0 < top_k_ratio <= 1 and positive finite alpha")
        self.base_layer = base_layer.requires_grad_(False)
        self.alpha = float(alpha)
        self.k = int(k) if k is not None else int(delta_init.numel() * top_k_ratio)
        spectrum = torch.fft.fft2(delta_init.detach().float())
        indices = topk_indices(spectrum.abs(), self.k)
        self.register_buffer("indices", indices)
        self.c = nn.Parameter(torch.view_as_real(spectrum.flatten()[indices]).clone())

    def delta_weight(self) -> torch.Tensor:
        values = torch.view_as_complex(self.c.contiguous())
        spectrum = values.new_zeros(self.base_layer.weight.numel()).scatter(0, self.indices, values)
        return self.alpha * torch.fft.ifft2(spectrum.reshape(self.base_layer.weight.shape)).real

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.delta_weight().to(dtype=self.base_layer.weight.dtype)
        return self.base_layer(x) + F.linear(x, delta)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Return an ordinary linear layer without mutating the adapter."""
        layer = copy.deepcopy(self.base_layer)
        layer.weight.add_(self.delta_weight().to(layer.weight.dtype))
        return layer
