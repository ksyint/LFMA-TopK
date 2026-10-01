from torch.utils.data import DataLoader

from experiments.tasks.definitions import BACKBONES
from .glue import TextCollator, read_glue
from .vision import ImageCollator, read_vision


def make_loader(config, processor, split):
    if split not in ('train', 'validation', 'test'):
        raise ValueError('split must be train, validation, or test')
    if BACKBONES[config['model']['backbone']]['family'] == 'vit':
        dataset, collator = read_vision(config, split), ImageCollator(processor)
    else:
        dataset, collator = read_glue(config, split), TextCollator(processor, config)
    return DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=split == 'train',
                      num_workers=config['data']['workers'], collate_fn=collator, pin_memory=True)
