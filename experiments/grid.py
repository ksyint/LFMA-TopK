"""Complete ViT/GLUE experiments with actual pretrained backbones."""
from copy import deepcopy
from itertools import product
from pathlib import Path

import yaml

from .tasks.definitions import BACKBONES, GLUE_TASKS, IMAGE_TASKS, validate_config

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'experiments' / 'configs' / 'catalog'
VISION_RATIOS = (0.0003, 0.05, 0.1)
NLU_RATIOS = {'roberta-base': (0.0016, 0.001, 0.0005), 'roberta-large': (0.0005, 0.0003, 0.0001)}
VISION_ALPHAS = (12.0, 120.0)
VISION_SEEDS = (42, 123, 456, 789, 1024)
NLU_SEEDS = (42, 123, 456, 789, 1024)


def ratio_name(value):
    return format(value, '.4f').replace('.', 'p')


def profile_key(config):
    backbone, task = config['model']['backbone'], config['data']['task']
    family = 'vision' if BACKBONES[backbone]['family'] == 'vit' else 'glue'
    return (Path(family) / backbone / task / f"ratio_{ratio_name(config['adapter']['top_k_ratio'])}" /
            f"alpha_{config['adapter']['alpha']:g}" / f"seed_{config['train']['seed']}")


def validate_profile(config):
    return validate_config(config)


def write_catalog():
    base = yaml.safe_load((ROOT / 'config.yaml').read_text())
    paths = []
    for backbone, spec in BACKBONES.items():
        vision = spec['family'] == 'vit'
        tasks, ratios, seeds = (IMAGE_TASKS, VISION_RATIOS, VISION_SEEDS) if vision else (GLUE_TASKS, NLU_RATIOS[backbone], NLU_SEEDS)
        alphas = VISION_ALPHAS if vision else (150.0,)
        for task, ratio, alpha, seed in product(tasks, ratios, alphas, seeds):
            config = deepcopy(base)
            config['model']['backbone'] = backbone
            config['data']['task'] = task
            config['adapter'].update(top_k_ratio=ratio, alpha=alpha)
            config['train'].update(seed=seed, epochs=50 if vision else 30,
                                   learning_rate=1e-4 if vision else 5e-2,
                                   head_learning_rate=1e-4 if vision else 1e-3)
            key = profile_key(config)
            config['train']['save_dir'] = str(Path('results/catalog') / key)
            validate_profile(config)
            path = CATALOG / key.with_suffix('.yaml')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# Pretrained LFMA experiment: backbone, benchmark, support ratio, scale, seed.\n' +
                            yaml.safe_dump(config, sort_keys=False))
            paths.append(path)
    return paths


def read_catalog():
    entries = []
    for path in sorted(CATALOG.rglob('*.yaml')):
        config = yaml.safe_load(path.read_text())
        entries.append((path, config, validate_profile(config)))
    if not entries:
        raise ValueError('Experiment catalog is empty')
    outputs = [config['train']['save_dir'] for _, config, _ in entries]
    if len(outputs) != len(set(outputs)):
        raise ValueError('Experiment output directories must be unique')
    return entries


if __name__ == '__main__':
    print(f'Wrote {len(write_catalog())} complete pretrained experiment profiles')
