"""Attach saved sparse supports to the exact pretrained encoder."""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from adapters.layers import FourierLinear
from experiments.tasks.backbones import load_backbone, target_names
from utils import cuda_device
from .metadata import read_metadata


def load_pretrained_adapter(directory, device='cuda', config=None):
    from safetensors.torch import load_file
    directory, device = Path(directory), cuda_device(device)
    metadata = read_metadata(directory)
    config = deepcopy(config or metadata['config'])
    model, processor = load_backbone(config, device, directory / 'processor')
    expected = set(target_names(model, config))
    if set(metadata['targets']) != expected:
        raise ValueError('Checkpoint target layers differ from the selected pretrained backbone')
    tensors = load_file(str(directory / 'adapter_model.safetensors'), device=str(device))
    modules = dict(model.named_modules())
    model.requires_grad_(False)
    for name, spec in metadata['targets'].items():
        base = modules[name]
        c, indices = tensors.pop(f'{name}.c'), tensors.pop(f'{name}.indices')
        if c.shape != (spec['k'], 2) or indices.shape != (spec['k'],):
            raise ValueError(f'Invalid coefficient/support shape at {name}')
        if indices.dtype != torch.int64 or indices.unique().numel() != len(indices):
            raise ValueError(f'Support indices must be unique int64 values at {name}')
        if indices.min() < 0 or indices.max() >= base.weight.numel():
            raise ValueError(f'Support indices exceed layer shape at {name}')
        layer = FourierLinear.__new__(FourierLinear)
        nn.Module.__init__(layer)
        layer.base_layer, layer.k, layer.alpha = base, spec['k'], spec['alpha']
        layer.register_buffer('indices', indices)
        layer.c = nn.Parameter(c)
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, layer)
    head = {key.removeprefix('classifier.'): value for key, value in tensors.items()
            if key.startswith('classifier.')}
    if len(head) != len(tensors):
        raise ValueError('Checkpoint contains unknown tensor keys')
    model.classifier.load_state_dict(head, strict=True)
    model.classifier.requires_grad_(True)
    return model, processor, config, metadata
