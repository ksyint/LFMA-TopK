#!/usr/bin/env python3
"""Evaluate and merge a Fourier adapter checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from adapters import merge_adapters
from adapters.io import load_adapted_model
from experiments.data import dataset
from experiments.metrics import evaluate


def main(args):
    torch.set_num_threads(1)
    model, state = load_adapted_model(args.checkpoint, args.device)
    model.eval()
    loader = DataLoader(dataset(state['config'], args.data, 'val'), batch_size=128)
    scores = evaluate(model, loader, args.device)
    merged = merge_adapters(model).eval()
    with torch.no_grad():
        x = next(iter(loader))[0].to(args.device)
        scores['merge_max_abs_error'] = (model(x) - merged(x)).abs().max().item()
        torch.testing.assert_close(model(x), merged(x), atol=1e-5, rtol=1e-5)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2) + '\n')
    print(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/eval.json')
    main(parser.parse_args())
