#!/usr/bin/env python3
"""Restore sparse adapters, evaluate real benchmarks, and optionally merge for deployment."""
import argparse
import json
from pathlib import Path

from adapters import merge_adapters
from adapters.io.pretrained import load_pretrained_adapter, read_metadata
from experiments.data.benchmarks import make_loader
from experiments.evaluation import write_merge_manifest
from experiments.runtime import apply_runtime_options
from experiments.optimization import run_epoch
from utils import cuda_device


def main(args):
    device = cuda_device(args.device)
    config = apply_runtime_options(read_metadata(args.checkpoint)['config'], args)
    model, processor, config, _ = load_pretrained_adapter(args.checkpoint, device, config)
    loader = make_loader(config, processor, args.split)
    scores, predictions = run_epoch(model, loader, device, config=config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(scores, indent=2) + '\n')
    (output / 'predictions.tsv').write_text('index\tprediction\n' + ''.join(f'{i}\t{value}\n' for i, value in enumerate(predictions)))
    if args.merge:
        merged = merge_adapters(model)
        merged.save_pretrained(args.merge, safe_serialization=True)
        processor.save_pretrained(args.merge)
        write_merge_manifest(args.merge, config, args.checkpoint)
    print(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--cache-dir')
    parser.add_argument('--model-dir')
    parser.add_argument('--data-root')
    parser.add_argument('--dataset-dir')
    parser.add_argument('--imagefolder')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/evaluation')
    parser.add_argument('--merge', help='Save a standard merged Transformers model in this directory')
    main(parser.parse_args())
