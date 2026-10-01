"""Explicit construction and checkpoint I/O; no global model registry."""
from pathlib import Path

import torch

from adapters import inject_adapters
from utils import cuda_device


def save_checkpoint(state, directory, is_best=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, directory / 'checkpoint_last.pth.tar')
    if is_best:
        torch.save(state, directory / 'checkpoint_best.pth.tar')


def load_adapted_model(path, device='cuda'):
    device = cuda_device(device)
    from model import make_model
    state = torch.load(path, map_location=device, weights_only=True)
    cfg = state['config']
    model = make_model(cfg).to(device)
    inject_adapters(model, seed=cfg['train']['seed'], **cfg['adapter'])
    model.load_state_dict(state['model'])
    return model, state
