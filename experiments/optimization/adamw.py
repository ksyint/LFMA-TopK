"""Separate learning rates for Fourier coefficients and the task head."""
import torch


def configure_optimizer(model, cfg):
    head_ids = {id(p) for p in model.classifier.parameters()} if hasattr(model, 'classifier') else set()
    adapter = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    groups = [{'params': adapter, 'lr': cfg['learning_rate']}]
    if head:
        groups.append({'params': head, 'lr': cfg['head_learning_rate']})
    return torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
