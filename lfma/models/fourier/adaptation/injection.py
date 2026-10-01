"""Named projection replacement and adapter-free model export."""

from __future__ import annotations
import copy
from typing import Dict, Iterable
import torch
from torch import nn
from lfma.models.fourier.adaptation.layers import FourierLinear


def inject_adapters(
    model: nn.Module,
    target_names: Iterable[str],
    top_k_ratio: float = 0.05,
    alpha: float = 12.0,
    init_std: float = 1e-3,
    seed: int = 42,
    probes: Dict[str, torch.Tensor] = None,
) -> Dict[str, FourierLinear]:
    """Freeze a model and replace exact named nn.Linear targets.

    Pass externally obtained spatial update probes to control initialization.
    Otherwise use seeded Gaussian spatial updates. A zero-sized floored support is an explicit error.
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
        delta = (
            probes[name]
            if probes is not None
            else torch.randn(layer.weight.shape, generator=generator) * init_std
        )
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
