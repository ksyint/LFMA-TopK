"""Configuration validation shared by all task families."""
import math

from experiments.protocols.models import BACKBONES
from experiments.protocols.datasets import GLUE_TASKS, GLUE_REVISION, IMAGE_TASKS, IMAGE_HUB

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
    return 2 * int(backbone['width'] ** 2 * ratio) * backbone['layers'] * (1 if backbone['family'] == 'vit' else 2)
