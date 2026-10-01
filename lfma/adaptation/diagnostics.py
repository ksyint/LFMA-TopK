"""Sparse coefficient and spatial update measurements for trained adapters."""

import torch

from lfma.adaptation.layers import FourierLinear


def adapter_layers(model):
    result = [
        (name, layer)
        for name, layer in model.named_modules()
        if isinstance(layer, FourierLinear)
    ]
    if not result:
        raise ValueError("No FourierLinear adapters were found")
    return result


def coefficient_statistics(layer):
    values = torch.view_as_complex(layer.c.detach().float().contiguous())
    magnitude = values.abs()
    energy = magnitude.square()
    total = energy.sum()
    probability = energy / total.clamp_min(torch.finfo(energy.dtype).tiny)
    nonzero = probability > 0
    entropy = -(probability[nonzero] * probability[nonzero].log()).sum()
    return {
        "coefficients": layer.k,
        "nonzero": int((magnitude > 0).sum()),
        "magnitude_mean": float(magnitude.mean()),
        "magnitude_maximum": float(magnitude.max()),
        "magnitude_median": float(magnitude.median()),
        "spectral_energy": float(total),
        "energy_entropy": float(entropy),
        "effective_coefficients": float(entropy.exp()) if total > 0 else 0.0,
    }


def radial_statistics(layer, bins=8):
    if bins < 1:
        raise ValueError("Radial bin count must be positive")
    rows, columns = layer.base_layer.weight.shape
    indices = layer.indices
    row = torch.div(indices, columns, rounding_mode="floor")
    column = indices.remainder(columns)
    vertical = torch.minimum(row, rows - row).float() / max(1, rows // 2)
    horizontal = torch.minimum(column, columns - column).float() / max(1, columns // 2)
    radius = (vertical.square() + horizontal.square()).sqrt() / 2**0.5
    assignments = (radius * bins).long().clamp(max=bins - 1)
    energy = layer.c.detach().float().square().sum(1)
    result = []
    for index in range(bins):
        selected = assignments == index
        result.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "coefficients": int(selected.sum()),
                "energy": float(energy[selected].sum()),
            }
        )
    return result


@torch.no_grad()
def spatial_statistics(layer, singular_values=False):
    delta = layer.delta_weight().float()
    base = layer.base_layer.weight.detach().float()
    norm = torch.linalg.vector_norm(delta)
    base_norm = torch.linalg.vector_norm(base)
    cosine = torch.nn.functional.cosine_similarity(
        delta.flatten(), base.flatten(), dim=0
    )
    result = {
        "shape": list(delta.shape),
        "norm": float(norm),
        "base_norm": float(base_norm),
        "relative_norm": float(norm / base_norm) if base_norm > 0 else None,
        "maximum_absolute_update": float(delta.abs().max()),
        "mean_update": float(delta.mean()),
        "base_cosine": float(cosine),
        "row_norm_mean": float(torch.linalg.vector_norm(delta, dim=1).mean()),
        "column_norm_mean": float(torch.linalg.vector_norm(delta, dim=0).mean()),
    }
    if singular_values:
        values = torch.linalg.svdvals(delta)
        energy = values.square()
        total = energy.sum()
        normalized = energy / total.clamp_min(torch.finfo(energy.dtype).tiny)
        cumulative = normalized.cumsum(0)
        result["stable_rank"] = float(total / energy[0]) if energy[0] > 0 else 0.0
        result["energy_rank_90"] = (
            int(torch.searchsorted(cumulative, 0.9).clamp(max=len(values) - 1)) + 1
            if total > 0
            else 0
        )
        result["top_singular_values"] = values[:16].cpu().tolist()
    return result


def gradient_statistics(model):
    rows = []
    for name, layer in adapter_layers(model):
        gradient = layer.c.grad
        if gradient is None:
            rows.append({"name": name, "present": False})
            continue
        values = gradient.detach().float()
        rows.append(
            {
                "name": name,
                "present": True,
                "finite": bool(torch.isfinite(values).all()),
                "norm": float(values.norm()),
                "maximum": float(values.abs().max()),
                "zero_fraction": float((values == 0).float().mean()),
            }
        )
    return rows


def support_overlap(left, right):
    first = {name: layer for name, layer in adapter_layers(left)}
    second = {name: layer for name, layer in adapter_layers(right)}
    if first.keys() != second.keys():
        raise ValueError("Support comparison requires matching projection names")
    rows = []
    for name in sorted(first):
        a, b = first[name], second[name]
        if a.base_layer.weight.shape != b.base_layer.weight.shape:
            raise ValueError("Support comparison requires matching matrix shapes")
        shared = int(torch.isin(a.indices, b.indices).sum())
        union = a.k + b.k - shared
        rows.append(
            {
                "name": name,
                "left": a.k,
                "right": b.k,
                "intersection": shared,
                "jaccard": shared / union,
                "left_coverage": shared / a.k,
                "right_coverage": shared / b.k,
            }
        )
    return rows


@torch.no_grad()
def inspect_model(model, singular_values=False):
    result = []
    for name, layer in adapter_layers(model):
        if layer.c.device.type != "cuda":
            raise ValueError("Spectral model diagnostics require CUDA")
        if not torch.isfinite(layer.c).all():
            raise ValueError(f"Nonfinite adapter coefficients at {name}")
        result.append(
            {
                "name": name,
                "alpha": layer.alpha,
                "coefficients": coefficient_statistics(layer),
                "radial": radial_statistics(layer),
                "spatial": spatial_statistics(layer, singular_values),
            }
        )
    return result
