"""Additional optimizer, sampler, and random state for exact epoch resume."""

import json
import os
import random
import tempfile
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _checkpoint_lock(directory):
    import fcntl

    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock = directory.parent / ('.' + directory.name + '.lock')
    with lock.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def recover_checkpoint(directory):
    directory = Path(directory)
    previous = directory.parent / ('.' + directory.name + '-previous')
    if not previous.exists():
        return directory
    with _checkpoint_lock(directory):
        if not directory.exists() and previous.is_dir():
            previous.replace(directory)
            _sync_directory(directory.parent)
    return directory


@contextmanager
def checkpoint_transaction(directory):
    directory = Path(directory)
    previous = directory.parent / ('.' + directory.name + '-previous')
    with _checkpoint_lock(directory):
        if directory.is_symlink() or previous.is_symlink():
            raise ValueError('Checkpoint destinations must be regular directories')
        if not directory.exists() and previous.exists():
            previous.replace(directory)
        staging = Path(tempfile.mkdtemp(
            prefix='.' + directory.name + '-staging-', dir=directory.parent
        ))
        moved_previous = False
        try:
            yield staging
            required = (
                'adapter_model.safetensors', 'adapter_config.json',
                'execution_state.pt', 'execution.json', 'inventory.json',
            )
            if any(not (staging / name).is_file() for name in required):
                raise ValueError('Checkpoint transaction is missing a required artifact')
            from lfma.artifacts.inventory import verify_inventory

            if not verify_inventory(staging)['valid']:
                raise ValueError('Checkpoint transaction failed its file inventory')
            for path in staging.rglob('*'):
                if path.is_file():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
            for path in sorted(staging.rglob('*'), reverse=True):
                if path.is_dir():
                    _sync_directory(path)
            _sync_directory(staging)
            if previous.exists():
                shutil.rmtree(previous)
            if directory.exists():
                directory.replace(previous)
                moved_previous = True
                _sync_directory(directory.parent)
            staging.replace(directory)
            _sync_directory(directory.parent)
        except BaseException:
            if moved_previous and not directory.exists():
                previous.replace(directory)
                _sync_directory(directory.parent)
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        if previous.exists():
            shutil.rmtree(previous)
            _sync_directory(directory.parent)


def random_state(device):
    if torch.device(device).type != "cuda":
        raise ValueError("Training state requires a CUDA device")
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def restore_random_state(state, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def optimizer_layout(engine):
    parameters = dict(engine.model.named_parameters())
    names = {id(parameter): name for name, parameter in parameters.items()}
    visited = set()
    layout = []
    for index, group in enumerate(engine.optimizer.param_groups):
        rows = []
        for parameter in group['params']:
            identity = id(parameter)
            if identity not in names or identity in visited:
                raise ValueError('Optimizer parameters must map uniquely to model tensors')
            if not parameter.requires_grad:
                raise ValueError('Optimizer contains a frozen backbone parameter')
            visited.add(identity)
            rows.append({
                'name': names[identity],
                'shape': list(parameter.shape),
                'dtype': str(parameter.dtype),
            })
        if not rows:
            raise ValueError('Optimizer groups must contain trainable tensors')
        layout.append({'group': group.get('name', str(index)), 'tensors': rows})
    expected = {id(parameter) for parameter in parameters.values()
                if parameter.requires_grad}
    if visited != expected:
        raise ValueError('Optimizer does not cover every trainable tensor')
    return layout


def validate_optimizer_state(state, layout, updates):
    groups = state.get('param_groups')
    values = state.get('state')
    if not isinstance(groups, list) or not isinstance(values, dict):
        raise ValueError('Execution checkpoint has an invalid optimizer structure')
    if len(groups) != len(layout):
        raise ValueError('Optimizer group count differs from its saved tensor layout')
    identities = set()
    for index, (group, specification) in enumerate(zip(groups, layout)):
        tensors = specification['tensors']
        if len(group['params']) != len(tensors):
            raise ValueError('Optimizer group tensor count changed')
        if group.get('name', str(index)) != specification['group']:
            raise ValueError('Optimizer group ordering changed')
        for identity, tensor in zip(group['params'], tensors):
            if identity in identities:
                raise ValueError('Optimizer state repeats a parameter identity')
            identities.add(identity)
            moments = values.get(identity)
            if moments is None:
                continue
            if not {'step', 'exp_avg', 'exp_avg_sq'} <= moments.keys():
                raise ValueError('An AdamW state is missing its update moments')
            for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if name not in moments:
                    continue
                value = moments[name]
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f'AdamW {name} must be a tensor')
                if list(value.shape) != tensor['shape']:
                    raise ValueError(f'Optimizer moment shape differs for {tensor["name"]}')
                if not torch.isfinite(value).all():
                    raise ValueError(f'Optimizer moment is nonfinite for {tensor["name"]}')
            step = moments['step']
            if isinstance(step, torch.Tensor):
                if step.numel() != 1:
                    raise ValueError('AdamW update coordinates must be scalar')
                step = step.item()
            if not np.isfinite(step) or int(step) != step or not 0 <= step <= updates:
                raise ValueError('AdamW update coordinate exceeds the training history')
    if values.keys() - identities:
        raise ValueError('Optimizer state contains tensors outside its parameter groups')


def save_execution(directory, engine, loader, epoch, best, config_sha256,
                   stopping=None, data_contract=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sampler = loader.sampler
    state = {
        "format": "lfma-execution-state-v1",
        "engine": engine.state_dict(),
        "optimizer": engine.optimizer.state_dict(),
        "optimizer_layout": optimizer_layout(engine),
        "random": random_state(engine.device),
        "sampler": sampler.state_dict() if hasattr(sampler, "state_dict") else None,
        "loader_generator": loader.generator.get_state()
        if loader.generator is not None
        else None,
        "epoch": int(epoch),
        "best": float(best),
        "config_sha256": config_sha256,
        "stopping": stopping,
        "data_contract": data_contract,
    }
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(directory / "execution_state.pt")
    marker = {
        "format": state["format"],
        "epoch": epoch,
        "best": best,
        "config_sha256": config_sha256,
        "optimizer_groups": len(engine.optimizer.param_groups),
        "precision": engine.precision.name,
        "updates": engine.precision.updates,
        "stopping": stopping,
        "data_contract": data_contract,
    }
    (directory / "execution.json").write_text(
        json.dumps(marker, indent=2, allow_nan=False) + "\n"
    )


def restore_execution(directory, engine, loader, config_sha256, data_contract=None):
    from lfma.artifacts.inventory import verify_inventory

    if not verify_inventory(directory)['valid']:
        raise ValueError('Execution checkpoint differs from its committed file inventory')
    path = Path(directory) / "execution_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "lfma-execution-state-v1":
        raise ValueError("Unsupported execution state format")
    if state["config_sha256"] != config_sha256:
        raise ValueError("Resume configuration differs from the recorded training run")
    if state.get("data_contract") != data_contract:
        raise ValueError("Training or validation inputs changed since this checkpoint")
    layout = optimizer_layout(engine)
    if state.get('optimizer_layout') != layout:
        raise ValueError('Optimizer parameter names, order, shapes, or dtypes changed')
    precision = state['engine']['precision']
    if precision['updates'] < 0 or precision['skipped'] < 0:
        raise ValueError('Optimizer update counts must be nonnegative')
    scheduler = state['engine']['schedule']
    if scheduler is not None and scheduler['updates'] != precision['updates']:
        raise ValueError('Scheduler and optimizer update coordinates differ')
    validate_optimizer_state(state['optimizer'], layout, precision['updates'])
    engine.optimizer.load_state_dict(state["optimizer"])
    engine.load_state_dict(state["engine"])
    if state["sampler"] is not None:
        if not hasattr(loader.sampler, "load_state_dict"):
            raise ValueError("Saved sampler state requires a resumable sampler")
        loader.sampler.load_state_dict(state["sampler"])
    if state["loader_generator"] is not None:
        if loader.generator is None:
            raise ValueError("Saved loader generator is missing")
        loader.generator.set_state(state["loader_generator"])
    restore_random_state(state["random"], engine.device)
    return state["epoch"], state["best"], state.get("stopping")


def execution_metadata(directory):
    path = Path(directory) / "execution.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("format") != "lfma-execution-state-v1":
        raise ValueError("Unsupported execution metadata")
    if value["epoch"] < 0 or value["updates"] < 0:
        raise ValueError("Execution epoch and update counts must be nonnegative")
    return value
