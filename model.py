"""Fixed-support Fourier adapters (LFMA, Sec. 3 and Algorithm 1)."""
from __future__ import annotations

import copy
import math
from typing import Dict, Iterable

import torch
from torch import nn
from torch.nn import functional as F


def topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    """Stable magnitude order; ties resolve to the lower flattened index."""
    if not 1 <= k <= values.numel():
        raise ValueError(f"k must lie in [1, {values.numel()}], got {k}")
    return torch.argsort(values.flatten(), descending=True, stable=True)[:k]


class FourierLinear(nn.Module):
    """W = W0 + alpha * Re(IFFT2(S(c))); exactly 2*k real adapter scalars.

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


def inject_adapters(model: nn.Module, target_names: Iterable[str],
                    top_k_ratio: float = 0.05, alpha: float = 12.0,
                    init_std: float = 1e-3, seed: int = 42,
                    probes: Dict[str, torch.Tensor] = None) -> Dict[str, FourierLinear]:
    """Freeze a model and replace exact named nn.Linear targets.

    Pass externally obtained spatial update probes to control initialization.
    Otherwise use seeded Gaussian updates: the manuscript does not prescribe
    their distribution. A zero-sized floored support is an explicit error.
    """
    names = list(target_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("target_names must be nonempty and unique")
    modules = dict(model.named_modules())
    for name in names:
        if name not in modules or not isinstance(modules[name], nn.Linear):
            raise ValueError(f"target {name!r} must name an nn.Linear")
    model.requires_grad_(False)
    result = {}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for name in names:
        layer = modules[name]
        delta = (probes[name] if probes is not None else
                 torch.randn(layer.weight.shape, generator=generator) * init_std)
        adapter = FourierLinear(layer, delta.to(layer.weight.device), top_k_ratio, alpha)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, adapter)
        result[name] = adapter
    return result


def merge_adapters(model: nn.Module) -> nn.Module:
    """Return an inference copy with every FourierLinear folded into its base."""
    result = copy.deepcopy(model)
    for name, module in list(result.named_modules()):
        if isinstance(module, FourierLinear):
            parent_name, _, child_name = name.rpartition(".")
            if not name:
                return module.merged()
            parent = result.get_submodule(parent_name) if parent_name else result
            setattr(parent, child_name, module.merged())
    return result
