"""Resolve pinned Hub/local artifacts and select the requested model family."""
import os
from pathlib import Path

from experiments.tasks.definitions import BACKBONES, num_labels
from utils import cuda_device


def hub_options(config):
    model = config['model']
    return {'cache_dir': model['cache_dir'], 'local_files_only': model['offline'],
            'revision': model['revision'] or BACKBONES[model['backbone']]['revision']}


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
        from .vision import load
    else:
        from .language import load
    model, processor = load(origin, options, num_labels(config), processor_dir)
    actual = (model.config.model_type, model.config.hidden_size, model.config.num_hidden_layers)
    expected = (spec['family'], spec['width'], spec['layers'])
    if actual != expected:
        raise ValueError(f'Pretrained directory has encoder {actual}, expected {expected}')
    if spec['family'] == 'vit' and (model.config.patch_size, model.config.image_size) != (16, 224):
        raise ValueError('ViT experiments require patch size 16 and image size 224')
    return model.to(device), processor
