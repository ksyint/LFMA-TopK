"""Describe a merged Transformers export and its LFMA provenance."""
import json
from pathlib import Path

from experiments.protocols.models import BACKBONES
from experiments.tasks.definitions import num_labels


def write_merge_manifest(directory, config, checkpoint):
    directory, checkpoint = Path(directory), Path(checkpoint)
    metadata = json.loads((checkpoint / 'adapter_config.json').read_text())
    spec = BACKBONES[config['model']['backbone']]
    manifest = {
        'format': 'lfma-merged-v1',
        'base_model': spec['id'],
        'revision': metadata['config']['model']['revision'],
        'backbone': config['model']['backbone'],
        'task': config['data']['task'],
        'num_labels': num_labels(config),
        'auto_class': 'AutoModelForImageClassification' if spec['family'] == 'vit' else 'AutoModelForSequenceClassification',
        'adapter': metadata['config']['adapter'],
        'adapted_projections': sorted(metadata['targets']),
        'selected_epoch': metadata['epoch'],
        'seed': config['train']['seed'],
        'files': [{'path': str(path.relative_to(directory)), 'bytes': path.stat().st_size}
                  for path in sorted(directory.rglob('*')) if path.is_file() and path.name != 'lfma_merge.json'],
    }
    (directory / 'lfma_merge.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
