"""Resume AdamW and CUDA random state while accepting explicit schedule overrides."""
from pathlib import Path

import torch


def restore_training_state(directory, optimizer, schedule, device):
    state = torch.load(Path(directory) / 'training_state.pt', map_location=device, weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    for group, rate in zip(optimizer.param_groups, (schedule['learning_rate'], schedule['head_learning_rate'])):
        group['lr'] = rate
        group['weight_decay'] = schedule['weight_decay']
    torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
