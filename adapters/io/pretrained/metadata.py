"""Read the portable LFMA checkpoint identity and training configuration."""
import json
from pathlib import Path


def read_metadata(directory):
    metadata = json.loads((Path(directory) / 'adapter_config.json').read_text())
    if metadata.get('format') != 'lfma-pretrained-v1':
        raise ValueError('Unsupported LFMA checkpoint format')
    return metadata
