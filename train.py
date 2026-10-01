#!/usr/bin/env python3
"""Train LFMA on ImageNet-21k ViT or RoBERTa using raw benchmark examples."""
import argparse
import json
from pathlib import Path

import torch

from adapters.io.pretrained import load_pretrained_adapter, read_metadata, save_pretrained_adapter
from experiments.data.benchmarks import make_loader
from experiments.tasks import GLUE_TASKS, validate_config
from experiments.tasks.backbones import insert_adapters, load_backbone
from experiments.optimization import configure_optimizer, run_epoch
from experiments.optimization.state import restore_training_state
from experiments.runtime import apply_runtime_options
from utils import cuda_device, load_config, set_seed





def main(args):
    device = cuda_device(args.device)
    if args.resume and args.config is None:
        config = read_metadata(args.resume)['config']
        # Apply dotted overrides through the same strict loader without temporary files.
        from utils import apply_overrides
        config = apply_overrides(config, args.opts)
    else:
        config = load_config(args.config or 'config.yaml', args.opts)
    config = apply_runtime_options(config, args)
    validate_config(config)
    set_seed(config['train']['seed'])
    torch.set_num_threads(config['train']['num_threads'])
    output = Path(config['train']['save_dir'])
    output.mkdir(parents=True, exist_ok=True)
    epoch, best = 0, float('-inf')
    if args.resume:
        saved = read_metadata(args.resume)
        for section in ('adapter',):
            if saved['config'][section] != config[section]:
                raise ValueError('Resume adapter configuration must match the checkpoint')
        if saved['config']['model']['backbone'] != config['model']['backbone'] or saved['config']['data']['task'] != config['data']['task']:
            raise ValueError('Resume backbone and task must match the checkpoint')
        config['model']['revision'] = saved['config']['model']['revision']
        model, processor, config, metadata = load_pretrained_adapter(args.resume, device, config)
        epoch, best = metadata['epoch'], metadata['best_score']
    else:
        model, processor = load_backbone(config, device)
        insert_adapters(model, config)
    if config['train']['epochs'] <= epoch:
        raise ValueError('Training epochs must exceed the resumed checkpoint epoch')
    optimizer = configure_optimizer(model, config['train'])
    if args.resume:
        restore_training_state(args.resume, optimizer, config['train'], device)
    train_loader, val_loader = (make_loader(config, processor, split) for split in ('train', 'validation'))
    primary = GLUE_TASKS.get(config['data']['task'], (None, None, 'accuracy'))[2]
    history_path = output / 'history.jsonl'
    with history_path.open('a' if args.resume else 'w') as history:
        for current in range(epoch, config['train']['epochs']):
            train_scores, _ = run_epoch(model, train_loader, device, optimizer, config)
            scores, _ = run_epoch(model, val_loader, device, config=config)
            improved = scores[primary] > best
            best = max(best, scores[primary])
            record = {'epoch': current + 1, 'train': train_scores, 'validation': scores}
            history.write(json.dumps(record) + '\n')
            history.flush()
            save_pretrained_adapter(model, processor, config, output / 'last', optimizer, current + 1, best)
            if improved:
                save_pretrained_adapter(model, processor, config, output / 'best', optimizer, current + 1, best)
            print(record, flush=True)
    counts = {'adapter_real_scalars': sum(p.numel() for n, p in model.named_parameters() if n.endswith('.c')),
              'head_parameters': sum(p.numel() for p in model.classifier.parameters()),
              'best_validation_metric': primary, 'best_validation_score': best, 'epoch': config['train']['epochs']}
    (output / 'metrics.json').write_text(json.dumps(counts, indent=2) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--opts', nargs='*', default=[])
    parser.add_argument('--resume', help='LFMA adapter checkpoint directory')
    parser.add_argument('--cache-dir')
    parser.add_argument('--model-dir', help='Local pretrained encoder directory')
    parser.add_argument('--data-root')
    parser.add_argument('--dataset-dir', help='Hugging Face DatasetDict saved with save_to_disk')
    parser.add_argument('--imagefolder', help='Local train/validation/test class folders')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
