"""Portable adapter state and completed experiment reports."""

import json
import torch
import csv
import statistics
from pathlib import Path
from copy import deepcopy
from lfma.adaptation.layers import FourierLinear
from lfma.core import (
    BACKBONES,
    load_backbone,
    target_names,
    cuda_device,
    num_labels,
    GLUE_TASKS,
)
from torch import nn
from collections import defaultdict


def read_metadata(directory):
    from lfma.artifacts.resume import recover_checkpoint

    directory = recover_checkpoint(directory)
    metadata = json.loads((Path(directory) / 'adapter_config.json').read_text())
    if metadata.get('format') != 'lfma-pretrained-v1':
        raise ValueError('Unsupported LFMA checkpoint format')
    return metadata


def save_pretrained_adapter(
    model, processor, config, directory, optimizer=None, epoch=0, best=None
):
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
    saved_config['model']['revision'] = (
        getattr(model.config, '_commit_hash', None)
        or config['model']['revision']
        or BACKBONES[config['model']['backbone']]['revision']
    )
    metadata = {
        'format': 'lfma-pretrained-v1',
        'config': saved_config,
        'targets': targets,
        'epoch': epoch,
        'best_score': best,
        'base_model': BACKBONES[config['model']['backbone']]['id'],
    }
    save_file(tensors, str(directory / 'adapter_model.safetensors'))
    (directory / 'adapter_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    processor.save_pretrained(directory / 'processor')
    if optimizer is not None:
        device = next(model.parameters()).device
        torch.save(
            {
                'optimizer': optimizer.state_dict(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state(device),
            },
            directory / 'training_state.pt',
        )


def load_pretrained_adapter(directory, device='cuda', config=None):
    from safetensors.torch import load_file

    directory, device = Path(directory), cuda_device(device)
    from lfma.artifacts.validation import validate_adapter

    metadata = read_metadata(directory)
    validate_adapter(directory)
    config = deepcopy(config or metadata['config'])
    model, processor = load_backbone(config, device, directory / 'processor')
    expected = set(target_names(model, config))
    if set(metadata['targets']) != expected:
        raise ValueError(
            'Checkpoint target layers differ from the selected pretrained backbone'
        )
    tensors = load_file(str(directory / 'adapter_model.safetensors'), device=str(device))
    modules = dict(model.named_modules())
    model.requires_grad_(False)
    for name, spec in metadata['targets'].items():
        base = modules[name]
        c, indices = tensors.pop(f'{name}.c'), tensors.pop(f'{name}.indices')
        if c.shape != (spec['k'], 2) or indices.shape != (spec['k'],):
            raise ValueError(f'Invalid coefficient/support shape at {name}')
        if indices.dtype != torch.int64 or indices.unique().numel() != len(indices):
            raise ValueError(f'Support indices must be unique int64 values at {name}')
        if indices.min() < 0 or indices.max() >= base.weight.numel():
            raise ValueError(f'Support indices exceed layer shape at {name}')
        layer = FourierLinear.__new__(FourierLinear)
        nn.Module.__init__(layer)
        layer.base_layer, layer.k, layer.alpha = base, spec['k'], spec['alpha']
        layer.register_buffer('indices', indices)
        layer.c = nn.Parameter(c)
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, layer)
    head = {
        key.removeprefix('classifier.'): value
        for key, value in tensors.items()
        if key.startswith('classifier.')
    }
    if len(head) != len(tensors):
        raise ValueError('Checkpoint contains unknown tensor keys')
    model.classifier.load_state_dict(head, strict=True)
    model.classifier.requires_grad_(True)
    return model, processor, config, metadata


def write_merge_manifest(directory, config, checkpoint):
    directory, checkpoint = Path(directory), Path(checkpoint)
    metadata = json.loads((checkpoint / 'adapter_config.json').read_text())
    spec = BACKBONES[config['model']['backbone']]
    manifest = {
        'format': 'lfma-merged-v1',
        'base_model': spec['id'],
        'revision': metadata['config']['model']['revision'],
        'backbone': config['model']['backbone'],
        'task': config['data']['task'],
        'num_labels': num_labels(config),
        'auto_class': (
            'AutoModelForImageClassification'
            if spec['family'] == 'vit'
            else 'AutoModelForSequenceClassification'
        ),
        'adapter': metadata['config']['adapter'],
        'adapted_projections': sorted(metadata['targets']),
        'selected_epoch': metadata['epoch'],
        'seed': config['train']['seed'],
        'files': [
            {'path': str(path.relative_to(directory)), 'bytes': path.stat().st_size}
            for path in sorted(directory.rglob('*'))
            if path.is_file() and path.name != 'lfma_merge.json'
        ],
    }
    (directory / 'lfma_merge.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def restore_training_state(directory, optimizer, schedule, device):
    state = torch.load(
        Path(directory) / 'training_state.pt', map_location=device, weights_only=True
    )
    optimizer.load_state_dict(state['optimizer'])
    for group, rate in zip(
        optimizer.param_groups, (schedule['learning_rate'], schedule['head_learning_rate'])
    ):
        group['lr'] = rate
        group['weight_decay'] = schedule['weight_decay']
    torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)


def collect_results(root, split='auto'):
    records = []
    for path in sorted(Path(root).rglob('best/adapter_config.json')):
        metadata = json.loads(path.read_text())
        config = metadata['config']
        task, run = config['data']['task'], path.parent.parent
        partition = (
            ('validation' if task in GLUE_TASKS else 'test') if split == 'auto' else split
        )
        primary = GLUE_TASKS.get(task, (None, None, 'accuracy'))[2]
        metrics_path = run / 'evaluation' / partition / 'metrics.json'
        if not metrics_path.is_file():
            continue
        scores = json.loads(metrics_path.read_text())
        if primary not in scores:
            continue
        records.append(
            {
                'backbone': config['model']['backbone'],
                'task': task,
                'ratio': config['adapter']['top_k_ratio'],
                'alpha': config['adapter']['alpha'],
                'seed': config['train']['seed'],
                'split': partition,
                'metric': primary,
                'score': scores[primary],
                'selected_epoch': metadata['epoch'],
                'scores': scores,
                'run': str(run),
            }
        )
    return records


def summarize_results(records, expected_seeds=(42, 123, 456, 789, 1024)):
    groups = defaultdict(list)
    keys = ('backbone', 'task', 'ratio', 'alpha', 'split', 'metric')
    for record in records:
        groups[tuple(record[key] for key in keys)].append(record)
    summary = []
    for key, entries in sorted(groups.items()):
        seeds = [entry['seed'] for entry in entries]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f'Duplicate seed results for {key}')
        scores = [entry['score'] for entry in entries]
        result = dict(zip(keys, key))
        result.update(
            runs=len(entries),
            seeds=sorted(seeds),
            missing_seeds=sorted(set(expected_seeds) - set(seeds)),
            median=statistics.median(scores),
            mean=statistics.mean(scores),
            std=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            minimum=min(scores),
            maximum=max(scores),
        )
        summary.append(result)
    return summary


def write_summary(records, output, expected_seeds=(42, 123, 456, 789, 1024)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(records, expected_seeds)
    (output / 'seed_records.json').write_text(json.dumps(records, indent=2) + '\n')
    (output / 'seed_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fields = [
        'backbone',
        'task',
        'ratio',
        'alpha',
        'split',
        'metric',
        'runs',
        'seeds',
        'missing_seeds',
        'median',
        'mean',
        'std',
        'minimum',
        'maximum',
    ]
    with (output / 'seed_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in summary:
            writer.writerow(
                {
                    name: json.dumps(value) if isinstance(value, list) else value
                    for name, value in result.items()
                }
            )
    return summary
