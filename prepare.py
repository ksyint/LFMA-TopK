"""Download pretrained artifacts or benchmark data into reusable local caches."""
import argparse
from pathlib import Path

from experiments.tasks import BACKBONES, GLUE_TASKS, IMAGE_TASKS
from experiments.tasks.definitions import IMAGE_HUB
from utils import load_config


def main(args):
    if args.backbone:
        from huggingface_hub import snapshot_download
        model = BACKBONES[args.backbone]
        path = snapshot_download(model['id'], revision=model['revision'], cache_dir=args.cache_dir,
                                 local_dir=args.local_dir, allow_patterns=['*.safetensors', 'config.json',
                                 'preprocessor_config.json', 'tokenizer*.json', 'vocab.json', 'merges.txt',
                                 'special_tokens_map.json'])
        print(f'Pretrained {args.backbone}: {path}')
    if args.task:
        config = load_config('config.yaml')
        config['data'].update(task=args.task, root=args.data_root, cache_dir=args.dataset_cache)
        if args.task in GLUE_TASKS:
            from experiments.data.benchmarks.hub import dataset_from_hub
            from experiments.tasks.definitions import GLUE_REVISION
            data = dataset_from_hub(config, 'nyu-mll/glue', args.task, GLUE_REVISION)
            if args.dataset_dir:
                data.save_to_disk(args.dataset_dir)
            print({split: len(records) for split, records in data.items()})
        elif args.task in IMAGE_HUB and args.dataset_dir:
            from datasets import DatasetDict
            from experiments.data.benchmarks.hub import dataset_from_hub
            repo, revision = IMAGE_HUB[args.task]
            splits = ('train', 'validation', 'test') if args.task == 'resisc45' else ('train', 'test')
            data = DatasetDict({split: dataset_from_hub(config, repo, revision=revision, split=split,
                               data_files={split: f'data/{split}-*.parquet'}) for split in splits})
            data.save_to_disk(args.dataset_dir)
            print({split: len(records) for split, records in data.items()})
        else:
            from experiments.data.benchmarks.vision import read_vision
            for split in ('train', 'validation', 'test'):
                dataset = read_vision(config, split)
                print(f'{args.task}/{split}: {len(dataset)} images')
    if not args.backbone and not args.task:
        raise ValueError('Select --backbone and/or --task')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=list(BACKBONES))
    parser.add_argument('--task', choices=list(GLUE_TASKS) + list(IMAGE_TASKS))
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--local-dir', help='Portable destination for pretrained model files')
    parser.add_argument('--data-root', default='datasets')
    parser.add_argument('--dataset-cache', default='.cache/datasets')
    parser.add_argument('--dataset-dir', help='Save a portable GLUE/Cars/RESISC45 DatasetDict here')
    main(parser.parse_args())
