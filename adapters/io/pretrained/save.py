"""Store sparse tensors, task head, processor, and resumable optimizer state."""
from copy import deepcopy
import json
from pathlib import Path

import torch

from adapters.layers import FourierLinear
from experiments.protocols.models import BACKBONES


def save_pretrained_adapter(model, processor, config, directory, optimizer=None, epoch=0, best=None):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors, targets = {}, {}
    for name, layer in model.named_modules():
        if isinstance(layer, FourierLinear):
            tensors[f'{name}.c'] = layer.c.detach().cpu().contiguous()
            tensors[f'{name}.indices'] = layer.indices.detach().cpu().contiguous()
            targets[name] = {'k': layer.k, 'alpha': layer.alpha}
    for name, tensor in model.classifier.state_dict().items():
        tensors[f'classifier.{name}'] = tensor.detach().cpu().contiguous()
    if not targets:
        raise ValueError('No LFMA adapters are installed')
    saved_config = deepcopy(config)
    saved_config['model']['revision'] = (getattr(model.config, '_commit_hash', None) or
                                         config['model']['revision'] or
                                         BACKBONES[config['model']['backbone']]['revision'])
    metadata = {'format': 'lfma-pretrained-v1', 'config': saved_config, 'targets': targets,
                'epoch': epoch, 'best_score': best,
                'base_model': BACKBONES[config['model']['backbone']]['id']}
    save_file(tensors, str(directory / 'adapter_model.safetensors'))
    (directory / 'adapter_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    processor.save_pretrained(directory / 'processor')
    if optimizer is not None:
        device = next(model.parameters()).device
        torch.save({'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(device)}, directory / 'training_state.pt')
