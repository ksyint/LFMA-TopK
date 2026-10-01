"""CUDA prediction on image paths or task-shaped JSONL text records."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from adapters.io.pretrained import load_pretrained_adapter, read_metadata
from experiments.data.benchmarks.glue import TextCollator
from experiments.tasks import BACKBONES
from experiments.runtime import apply_runtime_options
from utils import cuda_device


def main(args):
    device = cuda_device(args.device)
    config = apply_runtime_options(read_metadata(args.checkpoint)['config'], args)
    model, processor, config, _ = load_pretrained_adapter(args.checkpoint, device, config)
    model.eval()
    family = BACKBONES[config['model']['backbone']]['family']
    if family == 'vit':
        if not args.images:
            raise ValueError('Image models require --images')
        records = args.images
    else:
        if not args.text:
            raise ValueError('Text models require --text JSONL using task text fields')
        records = [json.loads(line) for line in Path(args.text).read_text().splitlines() if line.strip()]
    predictions = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            rows = records[start:start + args.batch_size]
            if family == 'vit':
                images = []
                for path in rows:
                    with Image.open(path) as image:
                        images.append(image.convert('RGB'))
                batch = processor(images=images, return_tensors='pt')
            else:
                batch = TextCollator(processor, config)(rows)
                batch.pop('labels', None)
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            values = logits.flatten().tolist() if config['data']['task'] == 'stsb' else logits.argmax(-1).tolist()
            predictions.extend(values)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'predictions': predictions}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images', nargs='+')
    parser.add_argument('--text')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model-dir')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/predictions.json')
    arguments = parser.parse_args()
    if arguments.batch_size < 1:
        parser.error('--batch-size must be positive')
    main(arguments)
