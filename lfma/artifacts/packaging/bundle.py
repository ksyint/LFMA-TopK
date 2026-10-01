"""Verified adapter archives for CUDA machines with prepared local backbones."""

import argparse
import io
import json
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryDirectory

from lfma.artifacts.packaging.validation import (
    artifact_inventory,
    validate_adapter,
    verify_integrity,
    write_integrity_manifest,
)


def normalized_member(name):
    value = Path(name)
    if value.is_absolute() or '..' in value.parts or not value.parts:
        raise ValueError(f'Unsafe archive member path: {name}')
    if any(part in ('', '.') for part in value.parts):
        raise ValueError(f'Archive member path is not normalized: {name}')
    return value


def archive_inventory(archive, maximum_bytes=50 * 1024**3):
    if maximum_bytes < 1:
        raise ValueError('Archive byte budget must be positive')
    rows, seen, total = [], set(), 0
    with tarfile.open(archive, 'r:*') as stream:
        for member in stream:
            relative = normalized_member(member.name)
            if str(relative) in seen:
                raise ValueError(f'Duplicate archive member: {relative}')
            seen.add(str(relative))
            if not member.isfile() and not member.isdir():
                raise ValueError(
                    'Adapter archives may only contain ordinary files and directories'
                )
            if member.size < 0:
                raise ValueError('Archive member has an invalid size')
            total += member.size
            if total > maximum_bytes:
                raise ValueError('Expanded archive exceeds the selected byte budget')
            rows.append(
                {
                    'path': str(relative),
                    'bytes': member.size,
                    'kind': 'file' if member.isfile() else 'directory',
                }
            )
    names = {row['path'] for row in rows if row['kind'] == 'file'}
    needed = {'adapter_config.json', 'adapter_model.safetensors', 'bundle_manifest.json'}
    if not needed <= names:
        raise ValueError(f'Adapter archive is missing {sorted(needed - names)}')
    return {'files': rows, 'expanded_bytes': total}


def _extract(archive, destination, maximum_bytes):
    inventory = archive_inventory(archive, maximum_bytes)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:*') as stream:
        for member in stream:
            path = destination / normalized_member(member.name)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise ValueError(f'Cannot read archive member: {member.name}')
            with source, path.open('xb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if path.stat().st_size != member.size:
                raise ValueError(f'Archive member size differs: {member.name}')
    verified = verify_integrity(destination)
    audit = validate_adapter(destination)
    return {'archive': inventory, 'integrity': verified, 'adapter': audit}


def _copy_checkpoint(source, destination, optimizer):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    validate_adapter(source)
    for row in artifact_inventory(source):
        relative = Path(row['path'])
        if relative.name == 'training_state.pt' and not optimizer:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    return write_integrity_manifest(destination)


def pack_adapter(checkpoint, archive, include_optimizer=False):
    archive = Path(archive).resolve()
    if archive.exists():
        raise FileExistsError(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='lfma-bundle-') as staging:
        directory = Path(staging)
        integrity = _copy_checkpoint(checkpoint, directory, include_optimizer)
        payload = {
            'format': 'lfma-portable-bundle-v1',
            'optimizer_included': bool(include_optimizer),
            'files': len(integrity['files']),
            'base_weights_embedded': False,
        }
        temporary = archive.with_name(archive.name + '.tmp')
        try:
            with tarfile.open(temporary, 'w:gz') as stream:
                for path in sorted(directory.rglob('*')):
                    if not path.is_file():
                        continue
                    name = str(path.relative_to(directory))
                    info = stream.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open('rb') as source:
                        stream.addfile(info, source)
            temporary.replace(archive)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    payload.update(archive=str(archive), archive_bytes=archive.stat().st_size)
    return payload


def unpack_adapter(archive, destination, maximum_bytes=50 * 1024**3):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Unpack into a new adapter directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.lfma-unpack-', dir=destination.parent) as staging:
        temporary = Path(staging) / 'adapter'
        report = _extract(archive, temporary, maximum_bytes)
        temporary.replace(destination)
    report['adapter']['directory'] = str(destination)
    report['integrity']['directory'] = str(destination)
    return report


def inspect_archive(archive, maximum_bytes=50 * 1024**3, verify=False):
    report = archive_inventory(archive, maximum_bytes)
    with tarfile.open(archive, 'r:*') as stream:
        source = stream.extractfile('adapter_config.json')
        if source is None:
            raise ValueError('Adapter metadata is missing')
        with source:
            metadata = json.load(io.TextIOWrapper(source, encoding='utf-8'))
    report.update(
        base_model=metadata['base_model'],
        revision=metadata['config']['model']['revision'],
        backbone=metadata['config']['model']['backbone'],
        task=metadata['config']['data']['task'],
        epoch=metadata['epoch'],
    )
    if verify:
        with TemporaryDirectory(prefix='lfma-verify-') as staging:
            report['verified'] = _extract(archive, staging, maximum_bytes)['integrity'][
                'verified_files'
            ]
    return report


def deployment_requirements(checkpoint):
    path = Path(checkpoint)
    metadata = json.loads((path / 'adapter_config.json').read_text())
    audit = validate_adapter(path)
    model = metadata['config']['model']
    required = {
        'adapter': ['adapter_config.json', 'adapter_model.safetensors', 'processor/'],
        'base_model': metadata['base_model'],
        'revision': model['revision'],
        'task': metadata['config']['data']['task'],
        'backbone': model['backbone'],
        'device': 'cuda',
        'processor_saved': True,
        'adapter_bytes': audit['bytes'],
        'download_command': [
            'hf',
            'download',
            metadata['base_model'],
            '--revision',
            model['revision'],
            '--local-dir',
            f'weights/{model["backbone"]}',
        ],
        'evaluation_command': [
            'python',
            'run.py',
            'evaluate',
            '--checkpoint',
            str(path),
            '--model-dir',
            f'weights/{model["backbone"]}',
            '--offline',
            '--device',
            'cuda',
        ],
    }
    return required


def bundle_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--pack', metavar='CHECKPOINT')
    operation.add_argument('--unpack', metavar='ARCHIVE')
    operation.add_argument('--inspect', metavar='ARCHIVE')
    operation.add_argument('--deployment', metavar='CHECKPOINT')
    parser.add_argument('--output')
    parser.add_argument('--include-optimizer', action='store_true')
    parser.add_argument('--maximum-gib', type=float, default=50)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    budget = int(args.maximum_gib * 1024**3)
    if args.pack:
        if not args.output:
            parser.error('--pack requires --output')
        result = pack_adapter(args.pack, args.output, args.include_optimizer)
    elif args.unpack:
        if not args.output:
            parser.error('--unpack requires --output')
        result = unpack_adapter(args.unpack, args.output, budget)
    elif args.inspect:
        result = inspect_archive(args.inspect, budget, args.verify)
    else:
        result = deployment_requirements(args.deployment)
    print(json.dumps(result, indent=2))
