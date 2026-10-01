"""CUDA training sessions, prediction export, and adapter inspection."""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import torch

from lfma.core import (
    BACKBONES,
    cuda_device,
    insert_adapters,
    load_backbone,
    load_config,
    set_seed,
    validate_config,
    apply_overrides,
)
from lfma.data.streams import make_loader
from lfma.data.descriptive import training_data_contract
from lfma.artifacts.storage import (
    load_pretrained_adapter,
    read_metadata,
    save_pretrained_adapter,
)
from lfma.artifacts.resume import (
    checkpoint_transaction, restore_execution, save_execution,
)
from lfma.artifacts.inventory import checkpoint_inventory, write_inventory
from lfma.adaptation.budgets import actual_budget
from lfma.adaptation.diagnostics import inspect_model
from lfma.adaptation.composition import (
    compatible_metadata,
    compose_models,
    normalized_weights,
)
from lfma.experiments.analysis import primary_metric, table_report
from lfma.experiments.prepare import save_json
from lfma.training.engine import EpochEngine
from lfma.training.journal import RunJournal, EarlyStopping, configuration_digest
from lfma.training.optimization import (
    build_optimizer,
    UpdateSchedule,
    update_budget,
    optimizer_summary,
)


def session_config(config):
    values = deepcopy(config)
    values["train"].setdefault("schedule", "constant")
    values["train"].setdefault("warmup_updates", 0)
    values["train"].setdefault("minimum_lr_ratio", 0.0)
    values["train"].setdefault("max_grad_norm", 1.0)
    values["train"].setdefault("patience", 0)
    values["train"].setdefault("minimum_improvement", 0.0)
    schedule = values["train"]
    if schedule["patience"] < 0 or schedule["minimum_improvement"] < 0:
        raise ValueError("Early-stopping patience and improvement must be nonnegative")
    if schedule["schedule"] not in ("constant", "linear", "cosine"):
        raise ValueError("Unknown learning-rate schedule")
    validate_config(values)
    return values


def resolve_config(args):
    if args.resume and args.config is None:
        config = apply_overrides(deepcopy(read_metadata(args.resume)['config']), args.opts)
    else:
        config = load_config(args.config or "config.yaml", args.opts)
    config = session_config(config)
    if args.device:
        cuda_device(args.device)
    return config


def write_checkpoint(
    directory, model, processor, config, engine, loader, epoch, best, fingerprint,
    stopping, data_contract, predictions, prediction_metadata,
):
    with checkpoint_transaction(directory) as staging:
        save_pretrained_adapter(model, processor, config, staging, epoch=epoch, best=best)
        save_execution(staging, engine, loader, epoch, best, fingerprint,
                       stopping, data_contract)
        predictions.save(staging / 'validation.npz', prediction_metadata)
        write_inventory(staging)


def fit_session(config, device="cuda", resume=None, diagnostics=False):
    config = session_config(config)
    device = cuda_device(device)
    set_seed(config["train"]["seed"])
    torch.set_num_threads(config["train"]["num_threads"])
    output = Path(config["train"]["save_dir"])
    epoch, best = 0, float("-inf")
    if resume:
        metadata = read_metadata(resume)
        saved = session_config(metadata["config"])
        # Loading pins an originally unspecified revision to its resolved commit.
        config["model"]["revision"] = saved["model"]["revision"]
        if configuration_digest(saved) != configuration_digest(config):
            raise ValueError("Resume requires the saved training configuration")
        model, processor, config, metadata = load_pretrained_adapter(
            resume, device, config
        )
        epoch, best = metadata["epoch"], metadata["best_score"]
    else:
        model, processor = load_backbone(config, device)
        config["model"]["revision"] = (
            getattr(model.config, "_commit_hash", None)
            or config["model"]["revision"]
            or BACKBONES[config["model"]["backbone"]]["revision"]
        )
        insert_adapters(model, config)
    if epoch >= config["train"]["epochs"]:
        raise ValueError(
            "The selected checkpoint has already completed the epoch budget"
        )
    train_loader = make_loader(config, processor, "train")
    validation_loader = make_loader(config, processor, "validation")
    data_contract = training_data_contract(
        train_loader, validation_loader, config['data']['task']
    )
    optimizer, groups = build_optimizer(model, config["train"])
    schedule = UpdateSchedule(
        optimizer,
        update_budget(train_loader, config["train"]),
        config["train"]["schedule"],
        config["train"]["warmup_updates"],
        config["train"]["minimum_lr_ratio"],
    )
    engine = EpochEngine(model, config, optimizer, schedule)
    fingerprint = configuration_digest(config)
    stopping_state = None
    if resume:
        restored_epoch, restored_best, stopping_state = restore_execution(
            resume, engine, train_loader, fingerprint, data_contract
        )
        if restored_epoch != epoch or restored_best != best:
            raise ValueError("Adapter and execution state belong to different epochs")
    journal = RunJournal(output, config, epoch if resume else None)
    save_json(output / "parameter-budget.json", actual_budget(model))
    save_json(output / "optimizer-groups.json", groups)
    save_json(output / 'data-contract.json', data_contract)
    metric = primary_metric(config["data"]["task"])
    stopping = EarlyStopping(config['train']['patience'],
                             config['train']['minimum_improvement'])
    stopping.restore(journal.rows, metric, stopping_state)
    if resume and stopping.best != best:
        raise ValueError('Checkpoint selection differs from validation history')
    if stopping.stopped:
        report = journal.summary(metric)
        report['stopping'] = stopping.state_dict()
        save_json(output / 'training-summary.json', report)
        return report
    for current in range(epoch, config["train"]["epochs"]):
        started = time.perf_counter()
        training, _ = engine.run(train_loader, current, training=True)
        validation, predictions = engine.run(validation_loader, current, collect=True)
        score = validation[metric]
        improved = stopping.update(score)
        best = stopping.best
        row = {
            "epoch": current + 1,
            "train": training,
            "validation": validation,
            "optimizer": optimizer_summary(optimizer),
            "wall_seconds": time.perf_counter() - started,
            'stopping': stopping.state_dict(),
        }
        prediction_metadata = {
            'split': 'validation', 'epoch': current + 1,
            'config_sha256': fingerprint,
            'dataset_sha256': data_contract['validation']['sha256'],
        }
        journal.append(row)
        if improved:
            write_checkpoint(
                output / "best",
                model,
                processor,
                config,
                engine,
                train_loader,
                current + 1,
                best,
                fingerprint,
                stopping.state_dict(), data_contract, predictions, prediction_metadata,
            )
            predictions.save(
                output / "validation-best.npz",
                prediction_metadata,
            )
        write_checkpoint(
            output / "last",
            model,
            processor,
            config,
            engine,
            train_loader,
            current + 1,
            best,
            fingerprint,
            stopping.state_dict(), data_contract, predictions, prediction_metadata,
        )
        predictions.save(output / 'validation-last.npz', prediction_metadata)
        print(json.dumps(row, allow_nan=False), flush=True)
        if stopping.stopped:
            break
    report = journal.summary(metric)
    report['stopping'] = stopping.state_dict()
    save_json(output / "training-summary.json", report)
    if diagnostics:
        save_json(output / "spectral-final.json", inspect_model(model))
    return report


def fit_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--opts", nargs="*", default=[])
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    report = fit_session(
        resolve_config(args), args.device, args.resume, args.diagnostics
    )
    print(json.dumps(report, indent=2))


def evaluate_detailed_cli():
    parser = argparse.ArgumentParser(
        description="Run a CUDA checkpoint and export aligned logits and task reports."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = deepcopy(read_metadata(args.checkpoint)["config"])
    if args.model_dir:
        config["model"]["local_dir"] = args.model_dir
    if args.manifest_dir:
        config["data"]["manifest_dir"] = args.manifest_dir
    if args.offline:
        config["model"]["offline"] = True
    device = cuda_device(args.device)
    model, processor, config, metadata = load_pretrained_adapter(
        args.checkpoint, device, config
    )
    loader = make_loader(config, processor, args.split)
    metrics, table = EpochEngine(model, config).run(loader, collect=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table.save(
        output / "predictions.npz",
        {
            "split": args.split,
            "epoch": metadata["epoch"],
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "revision": config["model"]["revision"],
            'dataset_sha256': metrics['dataset_sha256'],
        },
    )
    save_json(output / "metrics.json", metrics)
    save_json(output / "analysis.json", table_report(table, args.bootstrap, args.seed))
    table.export_jsonl(output / "predictions.jsonl", probabilities=True)
    print(json.dumps(metrics, indent=2))


def diagnostics_cli():
    parser = argparse.ArgumentParser(
        description="Measure sparse support and spatial updates of a CUDA adapter."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--singular-values", action="store_true")
    parser.add_argument("--verify-inventory", action="store_true")
    args = parser.parse_args()
    inventory = checkpoint_inventory(args.checkpoint, args.verify_inventory)
    if args.verify_inventory and not inventory["integrity"]["valid"]:
        raise ValueError("Checkpoint inventory differs from its files")
    model, _, _, _ = load_pretrained_adapter(args.checkpoint, cuda_device(args.device))
    report = {
        "checkpoint": inventory,
        "budget": actual_budget(model),
        "spectral": inspect_model(model, args.singular_values),
    }
    save_json(args.output, report)
    print(
        json.dumps(
            {"projections": len(report["spectral"]), "output": args.output}, indent=2
        )
    )


def compose_cli():
    parser = argparse.ArgumentParser(
        description="Compose compatible Fourier adapters on CUDA."
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head", choices=("first", "average"), default="average")
    parser.add_argument("--maximum-coefficients", type=int)
    parser.add_argument("--unnormalized", action="store_true")
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.weights):
        raise ValueError("Supply one weight for every checkpoint")
    metadata = [read_metadata(path) for path in args.checkpoints]
    config = compatible_metadata(metadata)
    weights = normalized_weights(args.weights, not args.unnormalized)
    device = cuda_device(args.device)
    models, processors = [], []
    for path in args.checkpoints:
        model, processor, _, _ = load_pretrained_adapter(path, device)
        models.append(model)
        processors.append(processor)
    composed, projections = compose_models(
        models, weights, args.maximum_coefficients, args.head
    )
    config["adapter"]["alpha"] = 1.0
    save_pretrained_adapter(composed, processors[0], config, args.output)
    save_json(
        Path(args.output) / "composition.json",
        {
            "checkpoints": [str(Path(path).resolve()) for path in args.checkpoints],
            "weights": weights,
            "head": args.head,
            "projections": projections,
        },
    )
    write_inventory(args.output)
    print(
        json.dumps({"output": args.output, "projections": len(projections)}, indent=2)
    )
