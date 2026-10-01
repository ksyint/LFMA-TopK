"""Exact attention-target selection and trainable task-head insertion."""
from torch import nn

from adapters import inject_adapters
from experiments.tasks.definitions import BACKBONES

def target_names(model, config):
    family = BACKBONES[config['model']['backbone']]['family']
    suffixes = ('.attention.attention.query',) if family == 'vit' else ('.attention.self.query', '.attention.self.value')
    names = [name for name, layer in model.named_modules()
             if isinstance(layer, nn.Linear) and name.endswith(suffixes)]
    expected = model.config.num_hidden_layers * len(suffixes)
    if len(names) != expected:
        raise ValueError(f'Expected {expected} {family} attention projections, found {len(names)}')
    return names

def insert_adapters(model, config):
    names = target_names(model, config)
    adapters = inject_adapters(model, names, seed=config['train']['seed'], **config['adapter'])
    # Classification/regression heads are task parameters, initialized with the task.
    model.classifier.requires_grad_(True)
    return adapters
