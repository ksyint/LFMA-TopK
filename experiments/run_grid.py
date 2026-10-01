"""Run filtered feature experiments through the ordinary root training script."""
import argparse
from pathlib import Path
import subprocess
import sys

from .grid import ROOT, read_catalog, profile_key


def matches(config, args):
    checks = ((args.feature_dims, config['model']['input_dim']),
              (args.ratios, config['adapter']['top_k_ratio']),
              (args.alphas, config['adapter']['alpha']),
              (args.seeds, config['train']['seed']))
    return all(values is None or current in values for values, current in checks)


def main(args):
    entries = read_catalog()
    selected = [entry for entry in entries if matches(entry[1], args)]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        selected = selected[:args.limit]
    if not selected:
        raise ValueError('No profile matches the requested experiment axes')
    print(f'Validated {len(entries)} profiles; selected {len(selected)} experiments.')
    if args.dry_run:
        for path, config, count in selected[:4]:
            print(f'{path.relative_to(ROOT)}: {count} real adapter parameters')
        return
    if not args.data_template or not args.base_template:
        raise ValueError('Runs require --data-template and --base-template')
    if args.device != 'cuda' and not args.device.startswith('cuda:'):
        raise ValueError('Experiment execution requires a CUDA device')
    for path, config, _ in selected:
        fields = {**config['model'], 'seed': config['train']['seed']}
        data = Path(args.data_template.format(**fields)).expanduser().resolve()
        base = Path(args.base_template.format(**fields)).expanduser().resolve()
        if not data.is_file() or not base.is_file():
            raise FileNotFoundError(f'Required inputs are missing: {data}, {base}')
        output = Path(args.output_root) / profile_key(config)
        command = [sys.executable, str(ROOT / 'train.py'), '--config', str(path),
                   '--data', str(data), '--base-checkpoint', str(base), '--device', args.device,
                   '--opts', 'train.save_dir', str(output)]
        if args.epochs is not None:
            command.extend(['train.epochs', str(args.epochs)])
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-dims', nargs='+', type=int)
    parser.add_argument('--ratios', nargs='+', type=float)
    parser.add_argument('--alphas', nargs='+', type=float)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--data-template', help='For example /data/features_{input_dim}.npz')
    parser.add_argument('--base-template', help='For example /weights/mlp_{input_dim}.pth')
    parser.add_argument('--output-root', default='results/catalog')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
