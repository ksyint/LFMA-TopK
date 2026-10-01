"""Feature-adapter experiment grid; every profile is a complete train.py config."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import math

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'experiments' / 'configs' / 'catalog'
FEATURE_DIMS = (384, 512, 768, 1024)
RATIOS = (0.0003, 0.001, 0.005, 0.01, 0.025, 0.05)
ALPHAS = (12.0, 60.0, 120.0)
SEEDS = (42, 123, 456)


def ratio_name(value):
    return format(value, '.4f').replace('.', 'p')


def profile_key(config):
    model, adapter, train = config['model'], config['adapter'], config['train']
    return (Path(f"feature_{model['input_dim']:04d}") /
            f"ratio_{ratio_name(adapter['top_k_ratio'])}" /
            f"alpha_{int(adapter['alpha']):03d}" / f"seed_{train['seed']}")


def validate_profile(config):
    model, adapter, train = config['model'], config['adapter'], config['train']
    if any(not isinstance(model[key], int) or model[key] < 1 for key in ('input_dim', 'hidden_dim', 'num_classes')):
        raise ValueError('Feature and class dimensions must be positive integers')
    if set(adapter['target_names']) != {'fc1', 'fc2'}:
        raise ValueError('Feature experiments adapt both fc1 and fc2')
    ratio = adapter['top_k_ratio']
    if not 0 < ratio <= 1 or not math.isfinite(adapter['alpha']) or adapter['alpha'] <= 0:
        raise ValueError('Invalid spectral ratio or scale')
    sizes = (model['input_dim'] * model['hidden_dim'], model['hidden_dim'] * model['num_classes'])
    if min(int(size * ratio) for size in sizes) < 1:
        raise ValueError('This profile allocates an empty Fourier support')
    if train['epochs'] < 1 or train['batch_size'] < 1 or train['learning_rate'] <= 0:
        raise ValueError('Invalid training schedule')
    return sum(2 * int(size * ratio) for size in sizes)


def write_catalog():
    base = yaml.safe_load((ROOT / 'config.yaml').read_text())
    paths = []
    for dimension, ratio, alpha, seed in product(FEATURE_DIMS, RATIOS, ALPHAS, SEEDS):
        config = deepcopy(base)
        config['model'].update(input_dim=dimension, hidden_dim=dimension, num_classes=1000)
        config['adapter'].update(top_k_ratio=ratio, alpha=alpha)
        config['train'].update(seed=seed, learning_rate=1e-4)
        key = profile_key(config)
        config['train']['save_dir'] = str(Path('results/catalog') / key)
        validate_profile(config)
        path = CATALOG / key.with_suffix('.yaml')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Feature-space LFMA: support ratio, scale, and seed.\n' + yaml.safe_dump(config, sort_keys=False))
        paths.append(path)
    return paths


def read_catalog():
    entries = []
    for path in sorted(CATALOG.rglob('*.yaml')):
        config = yaml.safe_load(path.read_text())
        parameters = validate_profile(config)
        entries.append((path, config, parameters))
    if not entries:
        raise ValueError('Experiment catalog is empty')
    destinations = [config['train']['save_dir'] for _, config, _ in entries]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Experiment output directories must be unique')
    return entries


if __name__ == '__main__':
    print(f'Wrote {len(write_catalog())} complete experiment profiles to {CATALOG}')
