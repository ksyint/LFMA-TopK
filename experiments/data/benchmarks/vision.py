"""Seven image benchmarks, canonical held-out splits, and checkpoint processors."""
from pathlib import Path

from torch.utils.data import Dataset, Subset

from experiments.tasks.definitions import IMAGE_HUB, num_labels
from .hub import dataset_from_hub
from .partitions import stratified_indices
from ..preprocessing import ImageCollator


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
        dataset = HubImages(dataset_from_hub(config, repo, revision=revision, split=actual, data_files=files))
        if task == 'resisc45' or split == 'test':
            return dataset
    elif task in ('cifar10', 'cifar100'):
        cls = datasets.CIFAR10 if task == 'cifar10' else datasets.CIFAR100
        dataset = cls(root, train=split != 'test', download=download)
        if split == 'test':
            return dataset
    elif task == 'oxford_pets':
        dataset = datasets.OxfordIIITPet(root, split='test' if split == 'test' else 'trainval',
                                        target_types='category', download=download)
        if split == 'test':
            return dataset
    elif task == 'fgvc_aircraft':
        return datasets.FGVCAircraft(root, split='val' if split == 'validation' else split,
                                     annotation_level='variant', download=download)
    elif task == 'eurosat':
        dataset = datasets.EuroSAT(root, download=download)
        groups = stratified_indices(labels_of(dataset), (0.7, 0.1, 0.2), seed)
        return Subset(dataset, groups[('train', 'validation', 'test').index(split)])
    else:
        raise ValueError(f'Unknown image task: {task}')
    groups = stratified_indices(labels_of(dataset), (1 - validation, validation), seed)
    return Subset(dataset, groups[0 if split == 'train' else 1])
