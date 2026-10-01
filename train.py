#!/usr/bin/env python3
"""Frozen-feature adaptation with root-level, explicit epoch functions.

python train.py --config config.yaml --opts train.epochs 50 adapter.alpha 12
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adapters import inject_adapters
from adapters.io import load_adapted_model, save_checkpoint
from experiments.data import dataset
from model import make_model
from utils import AverageMeter, cuda_device, load_config, set_seed


def configure_optimizer(model, cfg):
    parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_meter, accuracy_meter = AverageMeter(), AverageMeter()
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
            loss_meter.update(loss.item(), len(y))
            accuracy_meter.update((logits.argmax(-1) == y).float().mean().item(), len(y))
    return {'loss': loss_meter.avg, 'accuracy': accuracy_meter.avg, 'samples': loss_meter.count}


def main(args):
    args.device = cuda_device(args.device)
    cfg = load_config(args.config, args.opts)
    if args.smoke:
        cfg['train'].update(epochs=3, learning_rate=0.05, batch_size=64)
        cfg['adapter'].update(top_k_ratio=0.5, alpha=12.0)
    set_seed(cfg['train']['seed'])
    torch.set_num_threads(cfg['train'].get('num_threads', 1))
    start_epoch, best = 0, float('inf')
    if args.resume:
        model, saved = load_adapted_model(args.resume, args.device)
        if saved['config']['model'] != cfg['model'] or saved['config']['adapter'] != cfg['adapter']:
            raise ValueError('Resume model and adapter settings must match the saved checkpoint')
        start_epoch = saved['epoch']
        best = saved.get('best_loss', saved['metrics']['loss'])
    else:
        model = make_model(cfg).to(args.device)
        if args.base_checkpoint:
            model.load_state_dict(torch.load(args.base_checkpoint, map_location=args.device, weights_only=True))
        inject_adapters(model, seed=cfg['train']['seed'], **cfg['adapter'])
    optimizer = configure_optimizer(model, cfg['train'])
    if args.resume:
        optimizer.load_state_dict(saved['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = cfg['train']['learning_rate']
    train_loader = DataLoader(dataset(cfg, args.data, 'train'), batch_size=cfg['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset(cfg, args.data, 'val'), batch_size=128)
    initial = run_epoch(model, train_loader, args.device)
    output = Path(cfg['train']['save_dir'])
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(start_epoch, cfg['train']['epochs']):
        train_scores = run_epoch(model, train_loader, args.device, optimizer)
        scores = run_epoch(model, val_loader, args.device)
        is_best = scores['loss'] < best
        best = min(best, scores['loss'])
        history.append({'epoch': epoch + 1, 'train': train_scores, 'validation': scores})
        save_checkpoint({'config': cfg, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'epoch': epoch + 1, 'best_loss': best, 'metrics': scores}, output, is_best)
        print(f"Epoch {epoch + 1:03d} | train loss {train_scores['loss']:.4f} | val {scores}")
    final = run_epoch(model, train_loader, args.device)
    report = {'task': 'npz_features' if args.data else 'synthetic_sanity', 'initial_train': initial,
              'final_train': final, 'history': history,
              'trainable_real_scalars': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.smoke and final['loss'] >= initial['loss']:
        raise RuntimeError('Smoke optimization did not reduce training loss')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--opts', nargs='*', default=[])
    parser.add_argument('--data', help='NPZ feature dataset')
    parser.add_argument('--base-checkpoint', help='Unadapted FeatureMLP state_dict')
    parser.add_argument('--resume', help='Continue an adapter checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
