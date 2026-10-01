#!/usr/bin/env python3
"""Inspect architecture budgets and saved adapter headers without loading model tensors."""
import argparse
import json
from pathlib import Path

from experiments.evaluation.reports.parameters import parameter_budget, read_tensor_header
from utils import load_config


def main(args):
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
                raise ValueError(f'Checkpoint tensor count differs from configured architecture: {field}')
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--opts', nargs='*', default=[])
    parser.add_argument('--checkpoint')
    parser.add_argument('--output')
    main(parser.parse_args())
