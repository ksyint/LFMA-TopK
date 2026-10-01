"""Weighted unions of task-compatible sparse Fourier checkpoints."""

import math
from copy import deepcopy

import torch

from lfma.adaptation.layers import FourierLinear
from lfma.adaptation.diagnostics import adapter_layers


def compatible_metadata(metadata):
    if len(metadata) < 2:
        raise ValueError("Composition requires at least two adapter checkpoints")
    reference = metadata[0]
    for candidate in metadata[1:]:
        for section, field in (
            ("model", "backbone"),
            ("model", "revision"),
            ("data", "task"),
        ):
            if (
                reference["config"][section][field]
                != candidate["config"][section][field]
            ):
                raise ValueError(f"Adapter composition changed {section}.{field}")
        if set(reference["targets"]) != set(candidate["targets"]):
            raise ValueError("Composed adapters must target the same projections")
    return deepcopy(reference["config"])


def normalized_weights(values, normalize=True):
    weights = [float(value) for value in values]
    if len(weights) < 2 or not all(math.isfinite(value) for value in weights):
        raise ValueError("Composition requires at least two finite weights")
    if not any(value != 0 for value in weights):
        raise ValueError("Composition weights cannot all be zero")
    if normalize:
        total = sum(weights)
        if total == 0:
            raise ValueError("Normalized composition weights must have a nonzero sum")
        weights = [value / total for value in weights]
    return weights


@torch.no_grad()
def combine_coefficients(layers, weights, maximum=None):
    if len(layers) != len(weights):
        raise ValueError("Each adapter needs a composition weight")
    shape = layers[0].base_layer.weight.shape
    device = layers[0].c.device
    if device.type != "cuda":
        raise ValueError("Adapter composition requires CUDA")
    if any(layer.base_layer.weight.shape != shape for layer in layers):
        raise ValueError("Composed projection dimensions must match")
    support = torch.unique(
        torch.cat([layer.indices.to(device) for layer in layers]), sorted=True
    )
    values = torch.zeros(len(support), dtype=torch.complex64, device=device)
    for layer, weight in zip(layers, weights):
        positions = torch.searchsorted(support, layer.indices.to(device))
        coefficient = torch.view_as_complex(layer.c.detach().float().contiguous()).to(
            device
        )
        values.index_add_(0, positions, coefficient * (weight * layer.alpha))
    original_energy = values.abs().square().sum()
    if maximum is not None:
        if maximum < 1:
            raise ValueError("Composition coefficient limit must be positive")
        order = torch.argsort(values.abs(), descending=True, stable=True)[:maximum]
        support, values = support[order], values[order]
        order = torch.argsort(support)
        support, values = support[order], values[order]
    retained_energy = values.abs().square().sum()
    report = {
        "input_coefficients": [layer.k for layer in layers],
        "union_coefficients": int(
            torch.unique(
                torch.cat([layer.indices.to(device) for layer in layers])
            ).numel()
        ),
        "output_coefficients": len(support),
        "retained_energy": float(retained_energy / original_energy)
        if original_energy > 0
        else 1.0,
    }
    return support, values, report


@torch.no_grad()
def compose_models(models, weights, maximum=None, head="average"):
    if len(models) != len(weights):
        raise ValueError("Each model must have one weight")
    if head not in ("average", "first"):
        raise ValueError("Classifier composition must use average or first")
    verify_frozen_backbones(models)
    maps = [dict(adapter_layers(model)) for model in models]
    if any(mapping.keys() != maps[0].keys() for mapping in maps):
        raise ValueError("All models must expose identical target projection names")
    result = deepcopy(models[0])
    report = []
    for name in maps[0]:
        layers = [mapping[name] for mapping in maps]
        indices, values, row = combine_coefficients(layers, weights, maximum)
        old = result.get_submodule(name)
        combined = FourierLinear.__new__(FourierLinear)
        torch.nn.Module.__init__(combined)
        combined.base_layer = old.base_layer
        combined.alpha = 1.0
        combined.k = len(indices)
        combined.register_buffer("indices", indices)
        combined.c = torch.nn.Parameter(torch.view_as_real(values).contiguous())
        parent, _, child = name.rpartition(".")
        setattr(result.get_submodule(parent) if parent else result, child, combined)
        row['spatial_residual'] = composition_residual(layers, combined, weights)
        report.append(dict(name=name, **row))
    if head == "average":
        states = [model.classifier.state_dict() for model in models]
        merged = {}
        for key, tensor in states[0].items():
            if any(state[key].shape != tensor.shape for state in states):
                raise ValueError("Classifier tensor shapes differ between checkpoints")
            if tensor.is_floating_point():
                merged[key] = sum(
                    weight * state[key].float()
                    for weight, state in zip(weights, states)
                ).to(tensor.dtype)
            else:
                if any(not torch.equal(state[key], tensor) for state in states):
                    raise ValueError("Nonfloating classifier buffers must agree")
                merged[key] = tensor
        result.classifier.load_state_dict(merged)
    return result, report


@torch.no_grad()
def verify_frozen_backbones(models):
    if len(models) < 2:
        raise ValueError('Composition requires at least two models')
    reference = dict(models[0].named_parameters())
    excluded = {
        name for name in reference
        if name.startswith('classifier.') or name.endswith('.c')
    }
    expected = reference.keys() - excluded
    for index, model in enumerate(models[1:], start=1):
        candidate = dict(model.named_parameters())
        if candidate.keys() != reference.keys():
            raise ValueError(f'Composition input {index} has different parameter names')
        for name in sorted(expected):
            left, right = reference[name], candidate[name]
            if left.device.type != 'cuda' or right.device.type != 'cuda':
                raise ValueError('Frozen-backbone comparison requires CUDA')
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f'Frozen parameter layout differs at {name}')
            if not torch.equal(left, right.to(left.device)):
                raise ValueError(f'Composition inputs use different frozen weights at {name}')
        left_buffers = dict(models[0].named_buffers())
        right_buffers = dict(model.named_buffers())
        if left_buffers.keys() != right_buffers.keys():
            raise ValueError('Composition inputs expose different model buffers')
        for name, left in left_buffers.items():
            if name.endswith('.indices'):
                continue
            right = right_buffers[name]
            if left.shape != right.shape or not torch.equal(left, right.to(left.device)):
                raise ValueError(f'Frozen model buffer differs at {name}')


@torch.no_grad()
def composition_residual(reference_layers, combined, weights):
    expected = sum(
        weight * layer.delta_weight()
        for weight, layer in zip(weights, reference_layers)
    )
    residual = combined.delta_weight() - expected
    denominator = expected.square().sum().sqrt()
    return {
        "maximum_absolute_error": float(residual.abs().max()),
        "relative_norm_error": float(residual.norm() / denominator)
        if denominator > 0
        else float(residual.norm()),
    }
