"""Portable adapter validation from metadata and safetensors headers."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from lfma.models.fourier.core import BACKBONES, num_labels, validate_config

DTYPE_BYTES = {
    'F64': 8,
    'F32': 4,
    'F16': 2,
    'BF16': 2,
    'I64': 8,
    'I32': 4,
    'I16': 2,
    'I8': 1,
    'U8': 1,
    'BOOL': 1,
}


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_header(path):
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError('Truncated safetensors length header')
        length = struct.unpack('<Q', length_bytes)[0]
        if length < 2 or length > min(size - 8, 64 * 1024 * 1024):
            raise ValueError('Invalid safetensors JSON header length')
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError('Safetensors header must be a JSON object')
    total_bytes = size - length - 8
    intervals = []
    tensors = {}
    for name, entry in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(entry, dict) or entry.get('dtype') not in DTYPE_BYTES:
            raise ValueError(f'Unsupported tensor header for {name}')
        shape = entry.get('shape')
        offsets = entry.get('data_offsets')
        if not isinstance(shape, list) or any(
            type(value) is not int or value < 0 for value in shape
        ):
            raise ValueError(f'Invalid tensor shape for {name}')
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise ValueError(f'Invalid tensor byte range for {name}')
        first, last = offsets
        expected = math.prod(shape) * DTYPE_BYTES[entry['dtype']]
        if first < 0 or last < first or last > total_bytes or last - first != expected:
            raise ValueError(f'Tensor storage does not match shape and dtype: {name}')
        intervals.append((first, last, name))
        tensors[name] = {
            'dtype': entry['dtype'],
            'shape': shape,
            'elements': math.prod(shape),
            'bytes': expected,
        }
    cursor = 0
    for first, last, name in sorted(intervals):
        if first != cursor:
            raise ValueError(f'Tensor data contains overlap or gaps near {name}')
        cursor = last
    if cursor != total_bytes:
        raise ValueError('Safetensors file contains unaccounted payload bytes')
    return tensors


def expected_target_names(config):
    model = BACKBONES[config['model']['backbone']]
    names = []
    for index in range(model['layers']):
        if model['family'] == 'vit':
            names.append(f'vit.encoder.layer.{index}.attention.attention.query')
        else:
            for projection in ('query', 'value'):
                names.append(f'roberta.encoder.layer.{index}.attention.self.{projection}')
    return names


def expected_head_shapes(config):
    spec = BACKBONES[config['model']['backbone']]
    width, classes = spec['width'], num_labels(config)
    if spec['family'] == 'vit':
        return {'classifier.weight': [classes, width], 'classifier.bias': [classes]}
    return {
        'classifier.dense.weight': [width, width],
        'classifier.dense.bias': [width],
        'classifier.out_proj.weight': [classes, width],
        'classifier.out_proj.bias': [classes],
    }


def artifact_inventory(directory, include_hashes=False):
    directory = Path(directory).resolve()
    records = []
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.name == 'bundle_manifest.json':
            continue
        if path.is_symlink():
            raise ValueError(f'Portable artifacts must not use symlinks: {path}')
        row = {'path': str(path.relative_to(directory)), 'bytes': path.stat().st_size}
        if include_hashes:
            row['sha256'] = sha256_file(path)
        records.append(row)
    return records


def validate_adapter(directory, hashes=False):
    directory = Path(directory).resolve()
    metadata_path = directory / 'adapter_config.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('format') != 'lfma-pretrained-v1':
        raise ValueError('Unsupported LFMA adapter format')
    config = metadata['config']
    validate_config(config)
    spec = BACKBONES[config['model']['backbone']]
    if metadata['base_model'] != spec['id']:
        raise ValueError('Adapter base identity differs from the selected backbone')
    revision = config['model']['revision']
    if not isinstance(revision, str) or not revision:
        raise ValueError('Saved adapter must identify its base-model revision')
    expected = set(expected_target_names(config))
    if set(metadata['targets']) != expected:
        raise ValueError('Saved projection names differ from the configured transformer')
    if type(metadata['epoch']) is not int or metadata['epoch'] < 0:
        raise ValueError('Adapter epoch must be a nonnegative integer')
    if metadata.get('best_score') is not None and not math.isfinite(metadata['best_score']):
        raise ValueError('Best validation score must be finite')
    tensors = read_header(directory / 'adapter_model.safetensors')
    expected_shapes = expected_head_shapes(config)
    coefficient_total = 0
    for name, target in metadata['targets'].items():
        k, alpha = target['k'], target['alpha']
        if type(k) is not int or not 1 <= k <= spec['width'] ** 2:
            raise ValueError(f'Invalid sparse support count at {name}')
        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f'Invalid Fourier scale at {name}')
        expected_shapes[name + '.c'] = [k, 2]
        expected_shapes[name + '.indices'] = [k]
        coefficient_total += 2 * k
    if tensors.keys() != expected_shapes.keys():
        missing = sorted(expected_shapes.keys() - tensors.keys())
        extra = sorted(tensors.keys() - expected_shapes.keys())
        raise ValueError(f'Adapter tensor names differ. Missing {missing}, extra {extra}')
    for name, shape in expected_shapes.items():
        tensor = tensors[name]
        if tensor['shape'] != shape:
            raise ValueError(f'Unexpected tensor shape at {name}: {tensor["shape"]}')
        expected_dtype = 'I64' if name.endswith('.indices') else 'F32'
        if tensor['dtype'] != expected_dtype:
            raise ValueError(f'Expected {expected_dtype} at {name}, got {tensor["dtype"]}')
    processor = directory / 'processor'
    needed = 'preprocessor_config.json' if spec['family'] == 'vit' else 'tokenizer_config.json'
    if not (processor / needed).is_file():
        raise FileNotFoundError(processor / needed)
    files = artifact_inventory(directory, hashes)
    return {
        'format': 'lfma-adapter-audit-v1',
        'directory': str(directory),
        'backbone': config['model']['backbone'],
        'task': config['data']['task'],
        'base_model': metadata['base_model'],
        'revision': revision,
        'epoch': metadata['epoch'],
        'targets': len(expected),
        'adapter_real_scalars': coefficient_total,
        'head_parameters': sum(
            math.prod(shape)
            for name, shape in expected_shapes.items()
            if name.startswith('classifier.')
        ),
        'training_state': (directory / 'training_state.pt').is_file(),
        'files': files,
        'bytes': sum(row['bytes'] for row in files),
    }


def write_integrity_manifest(directory):
    directory = Path(directory)
    report = validate_adapter(directory, hashes=True)
    manifest = {'format': 'lfma-artifact-integrity-v1', 'files': report['files']}
    path = directory / 'bundle_manifest.json'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(path)
    return manifest


def verify_integrity(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'bundle_manifest.json').read_text())
    if manifest.get('format') != 'lfma-artifact-integrity-v1':
        raise ValueError('Unsupported artifact integrity manifest')
    seen = set()
    for row in manifest['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or row['path'] in seen:
            raise ValueError('Integrity manifest contains unsafe or duplicate paths')
        seen.add(row['path'])
        path = directory / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise ValueError(f'Artifact checksum differs: {path}')
    actual = {row['path'] for row in artifact_inventory(directory)}
    if actual != seen:
        raise ValueError('Artifact files differ from the integrity inventory')
    return {'verified_files': len(seen), 'directory': str(directory)}


def adapter_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--hashes', action='store_true')
    parser.add_argument('--write-integrity', action='store_true')
    parser.add_argument('--verify-integrity', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = validate_adapter(args.checkpoint, args.hashes)
    if args.write_integrity:
        report['integrity'] = write_integrity_manifest(args.checkpoint)
    if args.verify_integrity:
        report['verified'] = verify_integrity(args.checkpoint)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
