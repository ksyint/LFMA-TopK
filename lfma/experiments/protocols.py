"""Benchmark profiles, named protocols, and parameter inspection."""

import yaml
import argparse
import subprocess
import sys
import json
import math
import struct
from lfma.core import (
    BACKBONES,
    GLUE_TASKS,
    IMAGE_TASKS,
    validate_config,
    num_labels,
    load_config,
)
from copy import deepcopy
from itertools import product
from pathlib import Path
from pprint import pformat
from lfma.artifacts.storage import collect_results, write_summary

PROTOCOLS = {
    'table1': 'Both ViTs, seven image datasets, support 0.05, scale 12, five seeds',
    'vision-sparse': 'Both ViTs, seven image datasets, support 0.0003, scale 120, five seeds',
    'vision-ablation': 'Both ViTs, three supports, two scales, seven datasets, five seeds',
    'table2': 'RoBERTa-Base, six GLUE tasks, three supports, five seeds',
    'table3': 'RoBERTa-Large, six GLUE tasks, three supports, five seeds',
    'all': 'Every complete pretrained model/task/support/scale/seed profile',
}


def accepts(config, protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown experiment protocol: {protocol}')
    backbone = config['model']['backbone']
    vision = BACKBONES[backbone]['family'] == 'vit'
    ratio, alpha = config['adapter']['top_k_ratio'], config['adapter']['alpha']
    if protocol == 'table1':
        return vision and ratio == 0.05 and alpha == 12.0
    if protocol == 'vision-sparse':
        return vision and ratio == 0.0003 and alpha == 120.0
    if protocol == 'vision-ablation':
        return vision
    if protocol == 'table2':
        return backbone == 'roberta-base'
    if protocol == 'table3':
        return backbone == 'roberta-large'
    return True


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'experiments'
VISION_RATIOS = (0.0003, 0.05, 0.1)
NLU_RATIOS = {
    'roberta-base': (0.0016, 0.001, 0.0005),
    'roberta-large': (0.0005, 0.0003, 0.0001),
}
VISION_ALPHAS = (12.0, 120.0)
VISION_SEEDS = (42, 123, 456, 789, 1024)
NLU_SEEDS = (42, 123, 456, 789, 1024)


def ratio_name(value):
    return format(value, '.4f').replace('.', 'p')


def profile_key(config):
    backbone, task = config['model']['backbone'], config['data']['task']
    family = 'vision' if BACKBONES[backbone]['family'] == 'vit' else 'glue'
    return (
        Path(family)
        / backbone
        / task
        / f"ratio_{ratio_name(config['adapter']['top_k_ratio'])}"
        / f"alpha_{config['adapter']['alpha']:g}"
        / f"seed_{config['train']['seed']}"
    )


def validate_profile(config):
    return validate_config(config)


def catalog_path(key):
    family, backbone, task, ratio, alpha, seed = key.parts
    if (backbone, task, ratio, alpha) == (
        'vit-base',
        'cifar10',
        'ratio_0p0500',
        'alpha_12',
    ) and seed in ('seed_42', 'seed_123'):
        return CATALOG / f'{backbone}-{task}-{ratio}-{alpha}-{seed}'
    minimum = {
        'vit-base': 'ratio_0p0003',
        'vit-large': 'ratio_0p0003',
        'roberta-base': 'ratio_0p0005',
        'roberta-large': 'ratio_0p0001',
    }
    first_task = 'cifar10' if family == 'vision' else 'cola'
    first_alpha = 'alpha_12' if family == 'vision' else 'alpha_150'
    if (
        task == first_task
        and ratio == minimum[backbone]
        and alpha == first_alpha
        and seed in ('seed_1024', 'seed_123')
    ):
        return CATALOG / backbone / f'{task}-{ratio}-{alpha}-{seed}'
    folder = CATALOG / backbone / task
    if task == 'cifar10':
        if (ratio, alpha) != ('ratio_0p1000', 'alpha_12') or seed not in ('seed_42', 'seed_123'):
            folder /= ratio
            if alpha != 'alpha_120' or seed not in ('seed_42', 'seed_123'):
                folder /= alpha
    elif task == 'cola':
        largest = 'ratio_0p0016' if backbone == 'roberta-base' else 'ratio_0p0005'
        if ratio != largest or seed not in ('seed_42', 'seed_123'):
            folder /= ratio
    return folder / f'{ratio}-{alpha}-{seed}'


def write_catalog():
    base = load_config(ROOT / 'config.yaml')
    paths = []
    profiles = []
    for backbone, spec in BACKBONES.items():
        vision = spec['family'] == 'vit'
        tasks, ratios, seeds = (
            (IMAGE_TASKS, VISION_RATIOS, VISION_SEEDS)
            if vision
            else (GLUE_TASKS, NLU_RATIOS[backbone], NLU_SEEDS)
        )
        alphas = VISION_ALPHAS if vision else (150.0,)
        for task, ratio, alpha, seed in product(tasks, ratios, alphas, seeds):
            config = deepcopy(base)
            config['model']['backbone'] = backbone
            config['data']['task'] = task
            config['adapter'].update(top_k_ratio=ratio, alpha=alpha)
            config['train'].update(
                seed=seed,
                epochs=50 if vision else 30,
                learning_rate=1e-4 if vision else 5e-2,
                head_learning_rate=1e-4 if vision else 1e-3,
            )
            key = profile_key(config)
            config['train']['save_dir'] = str(Path('results/catalog') / key)
            validate_profile(config)
            profiles.append((key, config))
    python_keys = set(sorted(key for key, _ in profiles)[:96])
    for key, config in profiles:
        suffix = '.py' if key in python_keys else '.yaml'
        path = catalog_path(key).with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == '.py':
            source = 'cfg = ' + pformat(config, width=88, sort_dicts=False) + '\n'
        else:
            source = (
                '# Pretrained LFMA experiment: backbone, benchmark, support ratio, scale, seed.\n'
                + yaml.safe_dump(config, sort_keys=False)
            )
        path.write_text(source)
        path.with_suffix('.yaml' if suffix == '.py' else '.py').unlink(missing_ok=True)
        paths.append(path)
    return paths


def read_catalog():
    entries = []
    paths = (
        path
        for path in CATALOG.rglob('*')
        if path.is_file() and path.suffix in ('.yaml', '.yml', '.py')
    )
    for path in paths:
        config = load_config(path)
        entries.append((path, config, validate_profile(config)))
    entries.sort(key=lambda entry: profile_key(entry[1]).as_posix())
    keys = [profile_key(config) for _, config, _ in entries]
    if len(keys) != len(set(keys)):
        raise ValueError('Catalog profiles must have unique names across configuration formats')
    if not entries:
        raise ValueError('Experiment catalog is empty')
    outputs = [config['train']['save_dir'] for _, config, _ in entries]
    if len(outputs) != len(set(outputs)):
        raise ValueError('Experiment output directories must be unique')
    return entries


def catalog_cli():
    print(f'Wrote {len(write_catalog())} complete pretrained experiment profiles')


def matches(config, args):
    checks = (
        (args.backbones, config['model']['backbone']),
        (args.tasks, config['data']['task']),
        (args.ratios, config['adapter']['top_k_ratio']),
        (args.alphas, config['adapter']['alpha']),
        (args.seeds, config['train']['seed']),
    )
    return accepts(config, args.protocol) and all(
        values is None or current in values for values, current in checks
    )


def runtime_flags(config, args):
    flags = ['--device', args.device, '--data-root', str(Path(args.data_root).resolve())]
    if args.cache_dir:
        flags.extend(['--cache-dir', str(Path(args.cache_dir).resolve())])
    for attribute, flag, substitutions in [
        ('model_template', '--model-dir', {'backbone': config['model']['backbone']}),
        ('dataset_template', '--dataset-dir', {'task': config['data']['task']}),
        ('imagefolder_template', '--imagefolder', {'task': config['data']['task']}),
    ]:
        template = getattr(args, attribute)
        if template:
            flags.extend([flag, str(Path(template.format(**substitutions)).resolve())])
    if args.offline:
        flags.append('--offline')
    return flags


def run_grid_main(args):
    entries = read_catalog()
    selected = [entry for entry in entries if matches(entry[1], args)]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        selected = selected[: args.limit]
    if not selected:
        raise ValueError('No experiment matches the selected protocol and filters')
    print(f'Validated {len(entries)} profiles and selected {len(selected)} experiments.')
    if args.dry_run:
        for path, _, count in selected[:4]:
            print(f'{path.relative_to(ROOT)}: {count} real spectral parameters plus task head')
        return
    if args.device != 'cuda' and not args.device.startswith('cuda:'):
        raise ValueError('Experiment execution requires CUDA')
    if args.epochs is not None and args.epochs < 1:
        raise ValueError('--epochs must be positive')
    output_root = Path(args.output_root).resolve()
    for path, config, _ in selected:
        output = output_root / profile_key(config)
        flags = runtime_flags(config, args)
        command = [
            sys.executable,
            str(ROOT / 'run.py'),
            'train',
            '--config',
            str(path),
            *flags,
            '--opts',
            'train.save_dir',
            str(output),
        ]
        if args.epochs is not None:
            command.extend(['train.epochs', str(args.epochs)])
        subprocess.run(command, cwd=ROOT, check=True)
        if args.evaluate:
            split = 'validation' if config['data']['task'] in GLUE_TASKS else 'test'
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / 'run.py'),
                    'evaluate',
                    '--checkpoint',
                    str(output / 'best'),
                    '--split',
                    split,
                    '--output',
                    str(output / 'evaluation' / split),
                    *flags,
                ],
                cwd=ROOT,
                check=True,
            )


def argument_parser(evaluate=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', choices=PROTOCOLS, default='all')
    parser.add_argument('--backbones', nargs='+')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--ratios', nargs='+', type=float)
    parser.add_argument('--alphas', nargs='+', type=float)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--data-root', default='datasets')
    parser.add_argument('--cache-dir')
    parser.add_argument(
        '--model-template', help='Local pretrained directories, for example /weights/{backbone}'
    )
    parser.add_argument(
        '--dataset-template', help='Saved DatasetDict directories, for example /datasets/{task}'
    )
    parser.add_argument(
        '--imagefolder-template', help='ImageFolder directories, for example /images/{task}'
    )
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--output-root', default='results/catalog')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--evaluate', action=argparse.BooleanOptionalAction, default=evaluate)
    return parser


def grid_cli():
    run_grid_main(argument_parser().parse_args())


def parameter_budget(config):
    coefficients = validate_config(config)
    spec = BACKBONES[config['model']['backbone']]
    width, layers, classes = spec['width'], spec['layers'], num_labels(config)
    block = 12 * width * width + 13 * width
    if spec['family'] == 'vit':
        encoder = layers * block + 969 * width
        head = width * classes + classes
        projections = layers
    else:
        encoder = layers * block + 50782 * width
        head = width * width + width + width * classes + classes
        projections = 2 * layers
    total = encoder + head + coefficients
    return {
        'backbone': config['model']['backbone'],
        'task': config['data']['task'],
        'frozen_encoder_parameters': encoder,
        'head_parameters': head,
        'adapter_real_scalars': coefficients,
        'trainable_parameters': head + coefficients,
        'total_parameters': total,
        'trainable_fraction': (head + coefficients) / total,
        'adapted_projections': projections,
        'complex_coefficients_per_projection': int(
            width * width * config['adapter']['top_k_ratio']
        ),
    }


def read_tensor_header(path):
    path = Path(path)
    with path.open('rb') as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError('Invalid safetensors header length')
        length = struct.unpack('<Q', length_bytes)[0]
        if length > 16 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError('Invalid safetensors header size')
        metadata = json.loads(stream.read(length))
    tensors = {name: value for name, value in metadata.items() if name != '__metadata__'}
    counts = {'adapter_real_scalars': 0, 'head_parameters': 0, 'support_indices': 0}
    for name, spec in tensors.items():
        count = math.prod(spec['shape'])
        if name.endswith('.c'):
            counts['adapter_real_scalars'] += count
        elif name.endswith('.indices'):
            counts['support_indices'] += count
        elif name.startswith('classifier.'):
            counts['head_parameters'] += count
        else:
            raise ValueError(f'Unexpected adapter tensor: {name}')
    counts['trainable_parameters'] = counts['adapter_real_scalars'] + counts['head_parameters']
    return {
        'file': str(path),
        'bytes': path.stat().st_size,
        'counts': counts,
        'tensors': {
            name: {'shape': spec['shape'], 'dtype': spec['dtype']}
            for name, spec in tensors.items()
        },
    }


#!/usr/bin/env python3


def inspect_adapter_main(args):
    if args.checkpoint:
        directory = Path(args.checkpoint)
        config = json.loads((directory / 'adapter_config.json').read_text())['config']
    else:
        config = load_config(args.config, args.opts)
    report = {'architecture': parameter_budget(config)}
    if args.checkpoint:
        report['checkpoint'] = read_tensor_header(directory / 'adapter_model.safetensors')
        for field in ('adapter_real_scalars', 'head_parameters'):
            if report['architecture'][field] != report['checkpoint']['counts'][field]:
                raise ValueError(
                    f'Checkpoint tensor count differs from configured architecture: {field}'
                )
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text, end='')


def inspect_adapter_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--opts', nargs='*', default=[])
    parser.add_argument('--checkpoint')
    parser.add_argument('--output')
    inspect_adapter_main(parser.parse_args())


#!/usr/bin/env python3


def summarize_main(args):
    records = collect_results(args.root, args.split)
    if not records:
        raise ValueError(
            'No scored runs found. Run python run.py evaluate with --output <run>/evaluation/<split> first.'
        )
    summary = write_summary(records, args.output, args.expected_seeds)
    print(
        f'Wrote {len(summary)} task/setting summaries from {len(records)} evaluated runs to {args.output}'
    )


def summarize_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/catalog')
    parser.add_argument('--split', choices=['auto', 'validation', 'test'], default='auto')
    parser.add_argument(
        '--expected-seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024]
    )
    parser.add_argument('--output', default='results/summary')
    summarize_main(parser.parse_args())


#!/usr/bin/env python3


def run_protocol_cli():
    run_grid_main(argument_parser(evaluate=True).parse_args())
