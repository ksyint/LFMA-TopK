"""Count trainable scalars from pinned architecture metadata or a tensor header."""
import json
import math
from pathlib import Path
import struct

from experiments.protocols.models import BACKBONES
from experiments.tasks.definitions import num_labels, validate_config


def parameter_budget(config):
    coefficients = validate_config(config)
    spec = BACKBONES[config['model']['backbone']]
    width, layers, classes = spec['width'], spec['layers'], num_labels(config)
    block = 12 * width * width + 13 * width
    if spec['family'] == 'vit':
        encoder = layers * block + 969 * width
        head = width * classes + classes
        projections = layers
    else:
        encoder = layers * block + 50782 * width
        head = width * width + width + width * classes + classes
        projections = 2 * layers
    total = encoder + head + coefficients
    return {'backbone': config['model']['backbone'], 'task': config['data']['task'],
            'frozen_encoder_parameters': encoder, 'head_parameters': head,
            'adapter_real_scalars': coefficients, 'trainable_parameters': head + coefficients,
            'total_parameters': total, 'trainable_fraction': (head + coefficients) / total,
            'adapted_projections': projections,
            'complex_coefficients_per_projection': int(width * width * config['adapter']['top_k_ratio'])}


def read_tensor_header(path):
    path = Path(path)
    with path.open('rb') as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError('Invalid safetensors header length')
        length = struct.unpack('<Q', length_bytes)[0]
        if length > 16 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError('Invalid safetensors header size')
        metadata = json.loads(stream.read(length))
    tensors = {name: value for name, value in metadata.items() if name != '__metadata__'}
    counts = {'adapter_real_scalars': 0, 'head_parameters': 0, 'support_indices': 0}
    for name, spec in tensors.items():
        count = math.prod(spec['shape'])
        if name.endswith('.c'):
            counts['adapter_real_scalars'] += count
        elif name.endswith('.indices'):
            counts['support_indices'] += count
        elif name.startswith('classifier.'):
            counts['head_parameters'] += count
        else:
            raise ValueError(f'Unexpected adapter tensor: {name}')
    counts['trainable_parameters'] = counts['adapter_real_scalars'] + counts['head_parameters']
    return {'file': str(path), 'bytes': path.stat().st_size, 'counts': counts,
            'tensors': {name: {'shape': spec['shape'], 'dtype': spec['dtype']} for name, spec in tensors.items()}}
