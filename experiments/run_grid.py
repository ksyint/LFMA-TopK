"""Select complete pretrained profiles and run training with optional held-out scoring."""
import argparse
from pathlib import Path
import subprocess
import sys

from .grid import ROOT, read_catalog, profile_key
from .protocols.datasets import GLUE_TASKS
from .protocols.paper import PROTOCOLS, accepts


def matches(config, args):
    checks = ((args.backbones, config['model']['backbone']), (args.tasks, config['data']['task']),
              (args.ratios, config['adapter']['top_k_ratio']), (args.alphas, config['adapter']['alpha']),
              (args.seeds, config['train']['seed']))
    return accepts(config, args.protocol) and all(values is None or current in values for values, current in checks)


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


def main(args):
    entries = read_catalog()
    selected = [entry for entry in entries if matches(entry[1], args)]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        selected = selected[:args.limit]
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
        command = [sys.executable, str(ROOT / 'train.py'), '--config', str(path), *flags,
                   '--opts', 'train.save_dir', str(output)]
        if args.epochs is not None:
            command.extend(['train.epochs', str(args.epochs)])
        subprocess.run(command, cwd=ROOT, check=True)
        if args.evaluate:
            split = 'validation' if config['data']['task'] in GLUE_TASKS else 'test'
            subprocess.run([sys.executable, str(ROOT / 'eval.py'), '--checkpoint', str(output / 'best'),
                            '--split', split, '--output', str(output / 'evaluation' / split), *flags],
                           cwd=ROOT, check=True)


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
    parser.add_argument('--model-template', help='Local pretrained directories, for example /weights/{backbone}')
    parser.add_argument('--dataset-template', help='Saved DatasetDict directories, for example /datasets/{task}')
    parser.add_argument('--imagefolder-template', help='ImageFolder directories, for example /images/{task}')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--output-root', default='results/catalog')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--evaluate', action=argparse.BooleanOptionalAction, default=evaluate)
    return parser


if __name__ == '__main__':
    main(argument_parser().parse_args())
