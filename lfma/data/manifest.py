"""Portable JSONL benchmark partitions consumed by the native task collators."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from lfma.core import GLUE_TASKS, IMAGE_TASKS

SPLITS = ('train', 'validation', 'test')


def row_digest(row, task):
    if task in GLUE_TASKS:
        first, second, _ = GLUE_TASKS[task]
        content = {first: row[first]}
        if second:
            content[second] = row[second]
    else:
        content = {'image': str(Path(row['image']).resolve())}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def normalize_record(row, task, split, root, index, check_images=True):
    if task not in GLUE_TASKS and task not in IMAGE_TASKS:
        raise ValueError(f'Unknown benchmark task: {task}')
    if split not in SPLITS:
        raise ValueError(f'Unknown benchmark split: {split}')
    result = dict(row)
    result['id'] = str(row.get('id', f'{split}:{index}'))
    if not result['id']:
        raise ValueError('Record IDs must be nonempty')
    if task in GLUE_TASKS:
        first, second, _ = GLUE_TASKS[task]
        for field in (first, second):
            if field is not None and (
                not isinstance(row.get(field), str) or not row[field].strip()
            ):
                raise ValueError(f'{task} records require nonempty {field} text')
    else:
        if not isinstance(row.get('image'), str):
            raise ValueError('Image records require an image path')
        path = (Path(root) / row['image']).resolve()
        if check_images and not path.is_file():
            raise FileNotFoundError(path)
        result['image'] = str(path)
    label = row.get('label')
    if label in (None, '', -1, '-1'):
        if split != 'test' or task not in GLUE_TASKS:
            raise ValueError(
                'Labeled train/validation records and image test labels are required'
            )
        result['label'] = -1
    elif task == 'stsb':
        value = float(label)
        if not math.isfinite(value) or not 0 <= value <= 5:
            raise ValueError('STS-B labels must be finite scores in [0,5]')
        result['label'] = value
    else:
        value = int(label)
        if isinstance(label, bool) or float(label) != value:
            raise ValueError('Classification labels must be integer class IDs')
        classes = IMAGE_TASKS.get(task, 2)
        if not 0 <= value < classes:
            raise ValueError(f'{task} label must lie in [0,{classes})')
        result['label'] = value
    if 'group' in result:
        result['group'] = str(result['group'])
        if not result['group']:
            raise ValueError('Group IDs must be nonempty')
    return result


def read_partition(path, task, split, check_images=True):
    path = Path(path).resolve()
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as stream:
            values = list(csv.DictReader(stream))
    else:
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = []
    seen = set()
    for index, row in enumerate(values):
        normalized = normalize_record(row, task, split, path.parent, index, check_images)
        if normalized['id'] in seen:
            raise ValueError(f'Duplicate record ID in {path}: {normalized["id"]}')
        seen.add(normalized['id'])
        records.append(normalized)
    if not records:
        raise ValueError(f'Empty benchmark partition: {path}')
    return records


def partition_summary(records, task):
    labels = Counter(str(row['label']) for row in records)
    fingerprint = hashlib.sha256(
        json.dumps(records, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    result = {
        'records': len(records),
        'labels': dict(sorted(labels.items())),
        'sha256': fingerprint,
    }
    if task == 'stsb':
        scores = [row['label'] for row in records if row['label'] >= 0]
        result['label_range'] = [min(scores), max(scores)] if scores else None
    return result


def validate_partitions(partitions, task, check_duplicate_content=True):
    identities, groups, content = {}, {}, {}
    for split, rows in partitions.items():
        for row in rows:
            identity = row['id']
            if identities.setdefault(identity, split) != split:
                raise ValueError(f'Record ID occurs in multiple partitions: {identity}')
            if 'group' in row and groups.setdefault(row['group'], split) != split:
                raise ValueError(f'Group occurs in multiple partitions: {row["group"]}')
            if check_duplicate_content:
                key = row_digest(row, task)
                if content.setdefault(key, split) != split:
                    raise ValueError(f'Input content occurs in multiple partitions: {identity}')
    return {split: partition_summary(rows, task) for split, rows in partitions.items()}


def write_bundle(output, task, sources, check_duplicate_content=True):
    output = Path(output).resolve()
    if set(sources) != set(SPLITS):
        raise ValueError('Portable bundles require train, validation, and test sources')
    partitions = {split: read_partition(path, task, split) for split, path in sources.items()}
    summary = validate_partitions(partitions, task, check_duplicate_content)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, rows in partitions.items():
        filename = f'{split}.jsonl'
        temporary = output / (filename + '.tmp')
        temporary.write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        )
        temporary.replace(output / filename)
        paths[split] = filename
    metadata = {
        'format': 'lfma-dataset-manifest-v1',
        'task': task,
        'family': 'vision' if task in IMAGE_TASKS else 'text',
        'classes': IMAGE_TASKS.get(task, 1 if task == 'stsb' else 2),
        'splits': paths,
        'summary': summary,
        'check_duplicate_content': bool(check_duplicate_content),
    }
    temporary = output / 'manifest.json.tmp'
    temporary.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary.replace(output / 'manifest.json')
    return metadata


def read_bundle(directory, expected_task=None, check_images=True):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / 'manifest.json').read_text())
    if metadata.get('format') != 'lfma-dataset-manifest-v1':
        raise ValueError('Unsupported dataset manifest format')
    task = metadata['task']
    expected_family = 'vision' if task in IMAGE_TASKS else 'text'
    expected_classes = IMAGE_TASKS.get(task, 1 if task == 'stsb' else 2)
    if task not in GLUE_TASKS and task not in IMAGE_TASKS:
        raise ValueError('Dataset manifest names an unknown task')
    if metadata['family'] != expected_family or metadata['classes'] != expected_classes:
        raise ValueError('Dataset family or classifier size differs from its task')
    if expected_task is not None and task != expected_task:
        raise ValueError('Portable dataset task differs from the experiment configuration')
    if set(metadata['splits']) != set(SPLITS):
        raise ValueError('Dataset manifest requires train, validation, and test splits')
    partitions = {}
    for split, filename in metadata['splits'].items():
        path = Path(filename)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Partition paths must remain inside the dataset bundle')
        partitions[split] = read_partition(directory / path, task, split, check_images)
    summary = validate_partitions(
        partitions, task, metadata.get('check_duplicate_content', True)
    )
    if summary != metadata['summary']:
        raise ValueError('Dataset partition contents differ from the saved manifest')
    return metadata, partitions


class ManifestDataset(Dataset):
    def __init__(self, directory, task, split):
        if split not in SPLITS:
            raise ValueError('Unknown manifest split')
        metadata, partitions = read_bundle(directory, task)
        self.records = partitions[split]
        self.task = task
        self.metadata = metadata
        self.targets = [row['label'] for row in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if self.task in GLUE_TASKS:
            return dict(row)
        with Image.open(row['image']) as image:
            return image.convert('RGB'), row['label']


def manifest_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=sorted(set(GLUE_TASKS) | set(IMAGE_TASKS)))
    parser.add_argument('--train')
    parser.add_argument('--validation')
    parser.add_argument('--test')
    parser.add_argument('--output')
    parser.add_argument('--inspect')
    parser.add_argument('--allow-repeated-input', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        report, _ = read_bundle(args.inspect, args.task)
    else:
        if not all((args.task, args.train, args.validation, args.test, args.output)):
            parser.error('Supply --task, all three splits, and --output')
        sources = {split: getattr(args, split) for split in SPLITS}
        report = write_bundle(args.output, args.task, sources, not args.allow_repeated_input)
    print(json.dumps(report, indent=2))
