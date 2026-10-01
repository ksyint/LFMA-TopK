"""Pinned transformer architectures, model construction, and experiment settings."""

from __future__ import annotations
import torch
import copy
import math
import os
import random
import numpy as np
import yaml
from torch import nn
from torch.nn import functional as F
from typing import Dict, Iterable
from pathlib import Path


def topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    """Stable magnitude order. ties resolve to the lower flattened index."""
    if not 1 <= k <= values.numel():
        raise ValueError(f"k must lie in [1, {values.numel()}], got {k}")
    return torch.argsort(values.flatten(), descending=True, stable=True)[:k]


class FourierLinear(nn.Module):
    """W = W0 + alpha * Re(IFFT2(S(c))). exactly 2*k real adapter scalars.

    Shapes use PyTorch's [out_features, in_features] weight convention. The
    support is selected once from FFT2(delta_init), including its coefficients.
    No conjugate-symmetry constraint is imposed, as in the paper.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        delta_init: torch.Tensor,
        top_k_ratio: float = 0.05,
        alpha: float = 12.0,
        k: int = None,
    ):
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
        spectrum = values.new_zeros(self.base_layer.weight.numel()).scatter(
            0, self.indices, values
        )
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


BACKBONES = {
    'vit-base': {
        'family': 'vit',
        'id': 'google/vit-base-patch16-224-in21k',
        'revision': 'b4569560a39a0f1af58e3ddaf17facf20ab919b0',
        'width': 768,
        'layers': 12,
    },
    'vit-large': {
        'family': 'vit',
        'id': 'google/vit-large-patch16-224-in21k',
        'revision': '6074eaf2211423e928c93b93ef773d5da618aa7e',
        'width': 1024,
        'layers': 24,
    },
    'roberta-base': {
        'family': 'roberta',
        'id': 'FacebookAI/roberta-base',
        'revision': 'e2da8e2f811d1448a5b465c236feacd80ffbac7b',
        'width': 768,
        'layers': 12,
    },
    'roberta-large': {
        'family': 'roberta',
        'id': 'FacebookAI/roberta-large',
        'revision': '722cf37b1afa9454edce342e7895e588b6ff1d59',
        'width': 1024,
        'layers': 24,
    },
}


GLUE_TASKS = {
    'sst2': ('sentence', None, 'accuracy'),
    'mrpc': ('sentence1', 'sentence2', 'accuracy'),
    'qnli': ('question', 'sentence', 'accuracy'),
    'rte': ('sentence1', 'sentence2', 'accuracy'),
    'cola': ('sentence', None, 'matthews_correlation'),
    'stsb': ('sentence1', 'sentence2', 'pearson'),
}
GLUE_REVISION = 'bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c'


IMAGE_TASKS = {
    'cifar10': 10,
    'cifar100': 100,
    'oxford_pets': 37,
    'stanford_cars': 196,
    'fgvc_aircraft': 100,
    'eurosat': 10,
    'resisc45': 45,
}
IMAGE_HUB = {
    'resisc45': ('timm/resisc45', 'fe12fc5f1b7606543b0355eda392f1ddc54625c6'),
    'stanford_cars': ('tanganke/stanford_cars', '9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40'),
}


def num_labels(config):
    task = config['data']['task']
    return IMAGE_TASKS[task] if task in IMAGE_TASKS else (1 if task == 'stsb' else 2)


def validate_config(config):
    backbone = BACKBONES[config['model']['backbone']]
    task = config['data']['task']
    if task not in (IMAGE_TASKS if backbone['family'] == 'vit' else GLUE_TASKS):
        raise ValueError('Select an image task for ViT or a GLUE task for RoBERTa')
    ratio, scale = config['adapter']['top_k_ratio'], config['adapter']['alpha']
    if not 0 < ratio <= 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Spectral ratio and scale must be positive and finite')
    if int(backbone['width'] ** 2 * ratio) < 1:
        raise ValueError('The ratio allocates an empty Fourier support')
    schedule = config['train']
    for key in ('epochs', 'batch_size', 'gradient_accumulation'):
        if not isinstance(schedule[key], int) or schedule[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if schedule['learning_rate'] <= 0 or schedule['head_learning_rate'] <= 0:
        raise ValueError('Optimizer learning rates must be positive')
    if schedule['precision'] not in ('fp32', 'bf16'):
        raise ValueError('Choose fp32 or bf16 precision')
    if not 0 < config['data']['validation_fraction'] < 1:
        raise ValueError('validation_fraction must be strictly between zero and one')
    if config['data']['max_length'] > 512 or config['data']['max_length'] < 2:
        raise ValueError('RoBERTa maximum token length must be in [2, 512]')
    if config['data']['workers'] < 0:
        raise ValueError('workers must be nonnegative')
    return (
        2
        * int(backbone['width'] ** 2 * ratio)
        * backbone['layers']
        * (1 if backbone['family'] == 'vit' else 2)
    )


def vit_load(origin, options, classes, processor_dir=None):
    from transformers import AutoImageProcessor
    from transformers import AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(
        origin,
        num_labels=classes,
        ignore_mismatched_sizes=True,
        attn_implementation='eager',
        torch_dtype=torch.float32,
        **options,
    )
    processor_options = {'local_files_only': True} if processor_dir else dict(options)
    processor = AutoImageProcessor.from_pretrained(
        str(processor_dir or origin), use_fast=False, **processor_options
    )
    return model, processor


def roberta_load(origin, options, classes, processor_dir=None):
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        origin,
        num_labels=classes,
        ignore_mismatched_sizes=True,
        attn_implementation='eager',
        torch_dtype=torch.float32,
        **options,
    )
    tokenizer_options = {'local_files_only': True} if processor_dir else dict(options)
    tokenizer = AutoTokenizer.from_pretrained(str(processor_dir or origin), **tokenizer_options)
    return model, tokenizer


def hub_options(config):
    model = config['model']
    return {
        'cache_dir': model['cache_dir'],
        'local_files_only': model['offline'],
        'revision': model['revision'] or BACKBONES[model['backbone']]['revision'],
    }


def load_backbone(config, device='cuda', processor_dir=None):
    device = cuda_device(device)
    if config['model']['offline']:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    spec = BACKBONES[config['model']['backbone']]
    origin = config['model']['local_dir'] or spec['id']
    if config['model']['local_dir'] and not Path(origin).is_dir():
        raise FileNotFoundError(f'Pretrained model directory does not exist: {origin}')
    options = hub_options(config)
    if config['model']['local_dir']:
        options.pop('revision')
    if spec['family'] == 'vit':
        load = vit_load
    else:
        load = roberta_load
    model, processor = load(origin, options, num_labels(config), processor_dir)
    actual = (model.config.model_type, model.config.hidden_size, model.config.num_hidden_layers)
    expected = (spec['family'], spec['width'], spec['layers'])
    if actual != expected:
        raise ValueError(f'Pretrained directory has encoder {actual}, expected {expected}')
    if spec['family'] == 'vit' and (model.config.patch_size, model.config.image_size) != (
        16,
        224,
    ):
        raise ValueError('ViT experiments require patch size 16 and image size 224')
    return model.to(device), processor


def target_names(model, config):
    family = BACKBONES[config['model']['backbone']]['family']
    suffixes = (
        ('.attention.attention.query',)
        if family == 'vit'
        else ('.attention.self.query', '.attention.self.value')
    )
    names = [
        name
        for name, layer in model.named_modules()
        if isinstance(layer, nn.Linear) and name.endswith(suffixes)
    ]
    expected = model.config.num_hidden_layers * len(suffixes)
    if len(names) != expected:
        raise ValueError(
            f'Expected {expected} {family} attention projections, found {len(names)}'
        )
    return names


def insert_adapters(model, config):
    names = target_names(model, config)
    adapters = inject_adapters(model, names, seed=config['train']['seed'], **config['adapter'])
    # Classification/regression heads are task parameters, initialized with the task.
    model.classifier.requires_grad_(True)
    return adapters


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.total += float(value) * count
        self.count += count

    @property
    def avg(self):
        return self.total / max(1, self.count)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path, opts=None):
    with open(path, encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    return apply_overrides(cfg, opts)


def apply_overrides(cfg, opts=None):
    opts = opts or []
    if len(opts) % 2:
        raise ValueError('--opts expects pairs: dotted.key value')
    for key, value in zip(opts[::2], opts[1::2]):
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f'Unknown configuration key: {key}')
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def cuda_device(value='cuda'):
    device = torch.device(value)
    if device.type != 'cuda':
        raise ValueError('LFMA training and model evaluation require a CUDA device')
    return device
