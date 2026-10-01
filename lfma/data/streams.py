"""Raw vision and GLUE data with native pretrained preprocessing."""

import torch
import os
import numpy as np
from lfma.core import GLUE_TASKS, IMAGE_HUB, num_labels, GLUE_REVISION, BACKBONES
from pathlib import Path
from torch.utils.data import Dataset, Subset, DataLoader


class TextCollator:
    def __init__(self, tokenizer, config):
        self.tokenizer, self.config = tokenizer, config
        self.first, self.second, _ = GLUE_TASKS[config['data']['task']]

    def __call__(self, records):
        first = [record[self.first] for record in records]
        second = [record[self.second] for record in records] if self.second else None
        batch = self.tokenizer(
            first,
            second,
            truncation=True,
            padding=True,
            max_length=self.config['data']['max_length'],
            return_tensors='pt',
        )
        if all('label' in record and record['label'] >= 0 for record in records):
            dtype = torch.float32 if self.config['data']['task'] == 'stsb' else torch.long
            batch['labels'] = torch.tensor([record['label'] for record in records], dtype=dtype)
        return batch


class ImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, records):
        images, labels = zip(*records)
        batch = self.processor(
            images=[image.convert('RGB') for image in images], return_tensors='pt'
        )
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        return batch


def dataset_from_hub(config, repo_id, subset=None, revision=None, split=None, data_files=None):
    if config['model']['offline']:
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
    from datasets import DownloadConfig
    from datasets import load_dataset
    from datasets import load_from_disk

    data = config['data']
    if data['local_dir']:
        path = Path(data['local_dir'])
        if not path.is_dir():
            raise FileNotFoundError(f'Local dataset directory does not exist: {path}')
        result = load_from_disk(str(path))
        return result[split] if split else result
    return load_dataset(
        repo_id,
        subset,
        revision=revision,
        split=split,
        data_files=data_files,
        cache_dir=data['cache_dir'],
        download_config=DownloadConfig(local_files_only=config['model']['offline']),
    )


def stratified_indices(labels, fractions, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    groups = [[] for _ in fractions]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        cuts = np.cumsum([round(len(indices) * fraction) for fraction in fractions[:-1]])
        for group, values in zip(groups, np.split(indices, cuts)):
            group.extend(values.tolist())
    for group in groups:
        rng.shuffle(group)
        if not group:
            raise ValueError('Dataset split is empty. Adjust its fractions')
    return groups


def labels_of(dataset):
    for key in ('targets', '_labels'):
        if hasattr(dataset, key):
            return getattr(dataset, key)
    if hasattr(dataset, '_samples'):
        return [sample[1] for sample in dataset._samples]
    raise ValueError('Dataset does not expose classification targets')


class HubImages(Dataset):
    def __init__(self, records):
        self.records = records
        self.targets = records['label']

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return record['image'].convert('RGB'), record['label']


def read_vision(config, split):
    from torchvision import datasets

    task, data = config['data']['task'], config['data']
    root, download = data['root'], not config['model']['offline']
    validation, seed = data['validation_fraction'], data['split_seed']
    if data['imagefolder']:
        folder = Path(data['imagefolder']) / split
        dataset = datasets.ImageFolder(str(folder))
        reference = datasets.ImageFolder(str(Path(data['imagefolder']) / 'train'))
        if len(reference.classes) != num_labels(config):
            raise ValueError('ImageFolder class count must match the selected benchmark')
        if dataset.class_to_idx != reference.class_to_idx:
            raise ValueError('ImageFolder class names must match across train/validation/test')
        return dataset
    if task in IMAGE_HUB:
        repo, revision = IMAGE_HUB[task]
        actual = ('train' if split != 'test' else 'test') if task == 'stanford_cars' else split
        # Restrict Stanford Cars to original train/test parquet shards.
        files = {actual: f'data/{actual}-*.parquet'}
        dataset = HubImages(
            dataset_from_hub(config, repo, revision=revision, split=actual, data_files=files)
        )
        if task == 'resisc45' or split == 'test':
            return dataset
    elif task in ('cifar10', 'cifar100'):
        cls = datasets.CIFAR10 if task == 'cifar10' else datasets.CIFAR100
        dataset = cls(root, train=split != 'test', download=download)
        if split == 'test':
            return dataset
    elif task == 'oxford_pets':
        dataset = datasets.OxfordIIITPet(
            root,
            split='test' if split == 'test' else 'trainval',
            target_types='category',
            download=download,
        )
        if split == 'test':
            return dataset
    elif task == 'fgvc_aircraft':
        return datasets.FGVCAircraft(
            root,
            split='val' if split == 'validation' else split,
            annotation_level='variant',
            download=download,
        )
    elif task == 'eurosat':
        dataset = datasets.EuroSAT(root, download=download)
        groups = stratified_indices(labels_of(dataset), (0.7, 0.1, 0.2), seed)
        return Subset(dataset, groups[('train', 'validation', 'test').index(split)])
    else:
        raise ValueError(f'Unknown image task: {task}')
    groups = stratified_indices(labels_of(dataset), (1 - validation, validation), seed)
    return Subset(dataset, groups[0 if split == 'train' else 1])


def read_glue(config, split):
    task = config['data']['task']
    return dataset_from_hub(config, 'nyu-mll/glue', task, GLUE_REVISION, split)


def make_loader(config, processor, split):
    if split not in ('train', 'validation', 'test'):
        raise ValueError('split must be train, validation, or test')
    if config['data'].get('manifest_dir'):
        from lfma.data.manifest import ManifestDataset

        dataset = ManifestDataset(config['data']['manifest_dir'], config['data']['task'], split)
        collator = (
            ImageCollator(processor)
            if config['data']['task'] not in GLUE_TASKS
            else TextCollator(processor, config)
        )
    elif BACKBONES[config['model']['backbone']]['family'] == 'vit':
        dataset, collator = read_vision(config, split), ImageCollator(processor)
    else:
        dataset, collator = read_glue(config, split), TextCollator(processor, config)
    return DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=split == 'train',
        num_workers=config['data']['workers'],
        collate_fn=collator,
        pin_memory=True,
    )


def task_metrics(task, predictions, labels):
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import matthews_corrcoef

    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if task == 'stsb':
        if np.std(predictions) == 0 or np.std(labels) == 0 or len(labels) < 2:
            return {'pearson': 0.0, 'spearmanr': 0.0}
        return {
            'pearson': float(pearsonr(predictions, labels).statistic),
            'spearmanr': float(spearmanr(predictions, labels).statistic),
        }
    values = {'accuracy': float(accuracy_score(labels, predictions))}
    if task == 'cola':
        values['matthews_correlation'] = float(matthews_corrcoef(labels, predictions))
    if task == 'mrpc':
        values['f1'] = float(f1_score(labels, predictions, zero_division=0))
    return values
