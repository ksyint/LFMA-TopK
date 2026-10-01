"""Supervised transformer optimization and held-out prediction."""
from collections.abc import Mapping
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from experiments.tasks.metrics import task_metrics
from utils import AverageMeter


def run_epoch(model, loader, device, optimizer=None, config=None):
    config = config or {'data': {'task': 'classification'}, 'train': {'gradient_accumulation': 1, 'precision': 'fp32'}}
    training, task = optimizer is not None, config['data']['task']
    schedule = config['train']
    accumulation = schedule['gradient_accumulation']
    model.train(training)
    meter, predictions, references = AverageMeter(), [], []
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, item in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in item.items()} if isinstance(item, Mapping) else None
        labels = batch.get('labels') if batch is not None else item[1].to(device)
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if schedule['precision'] == 'bf16' else nullcontext()
        with torch.set_grad_enabled(training), amp:
            if batch is None:
                logits = model(item[0].to(device))
                loss = F.cross_entropy(logits, labels)
            else:
                output = model(**batch)
                logits, loss = output.logits, output.loss
            if training:
                if loss is None:
                    raise ValueError('Training examples must contain supervised labels')
                window_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
                (loss / window_size).backward()
        if training and ((step + 1) % accumulation == 0 or step + 1 == len(loader)):
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        estimate = logits.detach().flatten() if task == 'stsb' else logits.detach().argmax(-1)
        predictions.extend(estimate.float().cpu().tolist() if task == 'stsb' else estimate.cpu().tolist())
        if labels is not None:
            references.extend(labels.detach().cpu().tolist())
            meter.update(loss.detach().float().item(), len(labels))
    result = {'samples': len(predictions)}
    if references:
        result.update(loss=meter.avg, **task_metrics(task, predictions, references))
    return result, predictions
