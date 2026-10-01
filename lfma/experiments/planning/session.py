"""Recorded experiment plans with checkpoint-aware continuation and per-run logs."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

from lfma.experiments.catalog.protocols import ROOT, accepts, profile_key, read_catalog
from lfma.models.fourier.core import GLUE_TASKS, validate_config


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def select_profiles(protocol=None, backbones=None, tasks=None, seeds=None):
    selected = []
    for path, config, _ in read_catalog():
        if protocol and not accepts(config, protocol):
            continue
        filters = (
            (backbones, config['model']['backbone']),
            (tasks, config['data']['task']),
            (seeds, config['train']['seed']),
        )
        if all(choices is None or value in choices for choices, value in filters):
            selected.append((path, config))
    if not selected:
        raise ValueError('No catalog profiles match the plan filters')
    return selected


def create_plan(
    directory, profiles, epochs=None, offline=False, model_root=None, dataset_root=None
):
    directory = Path(directory).resolve()
    if (directory / 'plan.json').exists():
        raise FileExistsError('A plan already exists in the selected directory')
    directory.mkdir(parents=True, exist_ok=True)
    snapshots = directory / 'configs'
    snapshots.mkdir(exist_ok=True)
    entries, outputs = [], set()
    for source, original in profiles:
        config = deepcopy(original)
        if epochs is not None:
            config['train']['epochs'] = epochs
        config['model']['offline'] = bool(offline or config['model']['offline'])
        if model_root:
            config['model']['local_dir'] = str(
                (Path(model_root) / config['model']['backbone']).resolve()
            )
        if dataset_root:
            config['data']['local_dir'] = str(
                (Path(dataset_root) / config['data']['task']).resolve()
            )
        output = directory / 'runs' / profile_key(config)
        if output in outputs:
            raise ValueError(f'Two selected profiles use the same run directory: {output}')
        outputs.add(output)
        config['train']['save_dir'] = str(output)
        validate_config(config)
        identity = fingerprint(config)
        destination = snapshots / f'{identity}.yaml'
        destination.write_text(yaml.safe_dump(config, sort_keys=False))
        entries.append(
            {
                'id': identity,
                'source': str(Path(source).resolve()),
                'config': str(destination.relative_to(directory)),
                'output': str(output.relative_to(directory)),
                'backbone': config['model']['backbone'],
                'task': config['data']['task'],
                'seed': config['train']['seed'],
                'epochs': config['train']['epochs'],
            }
        )
    plan = {
        'format': 'lfma-experiment-plan-v1',
        'created': datetime.now(timezone.utc).isoformat(),
        'runs': entries,
    }
    atomic_json(directory / 'plan.json', plan)
    return plan


class PlanSession:
    def __init__(self, directory, device='cuda'):
        self.directory = Path(directory).resolve()
        self.plan = json.loads((self.directory / 'plan.json').read_text())
        if self.plan.get('format') != 'lfma-experiment-plan-v1':
            raise ValueError('Unsupported LFMA experiment plan')
        if device != 'cuda' and not device.startswith('cuda:'):
            raise ValueError('Experiment plans execute models on CUDA')
        self.device = device
        self.status_path = self.directory / 'status.json'
        self.status = (
            json.loads(self.status_path.read_text()) if self.status_path.exists() else {}
        )
        identities = [row['id'] for row in self.plan['runs']]
        if len(identities) != len(set(identities)):
            raise ValueError('Plan run IDs must be unique')
        if set(self.status) - set(identities):
            raise ValueError('Plan status contains unknown run IDs')
        for entry in self.plan['runs']:
            path = self._path(entry['config'])
            config = yaml.safe_load(path.read_text())
            if fingerprint(config) != entry['id']:
                raise ValueError(f'Configuration snapshot changed: {path}')
            if Path(config['train']['save_dir']).resolve() != self._path(entry['output']):
                raise ValueError('Plan output differs from its resolved configuration')

    def _path(self, relative):
        value = Path(relative)
        if value.is_absolute() or '..' in value.parts:
            raise ValueError('Plan files must remain inside their session directory')
        return self.directory / value

    def _record(self, entry, state, **details):
        event = {'state': state, 'updated': datetime.now(timezone.utc).isoformat(), **details}
        self.status[entry['id']] = event
        atomic_json(self.status_path, self.status)
        with (self.directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'run': entry['id'], **event}) + '\n')

    def _command(self, command, output, label):
        output.mkdir(parents=True, exist_ok=True)
        with (output / f'{label}.log').open('a') as stream:
            stream.write('\n' + json.dumps(command) + '\n')
            stream.flush()
            subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True
            )

    def _one(self, entry, evaluate=True):
        output = self._path(entry['output'])
        config = self._path(entry['config'])
        checkpoint = output / 'last'
        metadata = checkpoint / 'adapter_config.json'
        completed_epochs = 0
        if metadata.is_file():
            saved = json.loads(metadata.read_text())
            resolved = yaml.safe_load(config.read_text())
            for key in ('adapter', 'data'):
                if saved['config'][key] != resolved[key]:
                    raise ValueError(f'Resume checkpoint {key} differs from the plan')
            if saved['config']['model']['backbone'] != entry['backbone']:
                raise ValueError('Resume checkpoint backbone differs from the plan')
            completed_epochs = saved['epoch']
        if completed_epochs < entry['epochs']:
            command = [
                sys.executable,
                str(ROOT / 'run.py'),
                'train',
                '--config',
                str(config),
                '--device',
                self.device,
            ]
            if completed_epochs:
                command.extend(['--resume', str(checkpoint)])
            self._record(entry, 'training', resumed_epoch=completed_epochs)
            self._command(command, output, 'train')
        from lfma.artifacts.adapter.validation import validate_adapter

        audit = validate_adapter(output / 'last')
        if audit['epoch'] != entry['epochs']:
            raise ValueError('Completed checkpoint epoch differs from the planned schedule')
        if evaluate:
            split = 'validation' if entry['task'] in GLUE_TASKS else 'test'
            evaluation = output / 'evaluation' / split
            self._record(entry, 'evaluating', epoch=audit['epoch'])
            command = [
                sys.executable,
                str(ROOT / 'run.py'),
                'evaluate',
                '--checkpoint',
                str(output / 'best'),
                '--split',
                split,
                '--output',
                str(evaluation),
                '--device',
                self.device,
            ]
            self._command(command, output, 'evaluate')
            scores = json.loads((evaluation / 'metrics.json').read_text())
        else:
            scores = None
        self._record(
            entry,
            'complete',
            epoch=audit['epoch'],
            metrics=scores,
            adapter_real_scalars=audit['adapter_real_scalars'],
        )

    def run(self, evaluate=True, keep_going=False):
        import fcntl

        with (self.directory / 'session.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    'Another process is executing this experiment plan'
                ) from error
            for entry in self.plan['runs']:
                state = self.status.get(entry['id'], {})
                if state.get('state') == 'complete' and (
                    not evaluate or state.get('metrics') is not None
                ):
                    continue
                try:
                    self._one(entry, evaluate)
                except (OSError, ValueError, subprocess.CalledProcessError) as error:
                    self._record(entry, 'failed', error=str(error))
                    if not keep_going:
                        raise
        return self.summary()

    def summary(self):
        counts = {}
        for entry in self.plan['runs']:
            state = self.status.get(entry['id'], {}).get('state', 'pending')
            counts[state] = counts.get(state, 0) + 1
        return {
            'runs': len(self.plan['runs']),
            'states': counts,
            'directory': str(self.directory),
        }


def plan_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--create', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--protocol')
    parser.add_argument('--backbones', nargs='+')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--model-root')
    parser.add_argument('--dataset-root')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.create:
        profiles = select_profiles(args.protocol, args.backbones, args.tasks, args.seeds)
        create_plan(
            args.directory,
            profiles,
            args.epochs,
            args.offline,
            args.model_root,
            args.dataset_root,
        )
    session = PlanSession(args.directory, args.device)
    result = (
        session.run(not args.train_only, args.keep_going) if args.execute else session.summary()
    )
    print(json.dumps(result, indent=2))
