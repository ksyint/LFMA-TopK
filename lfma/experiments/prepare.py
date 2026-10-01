"""Dataset inspection and parameter-budget preparation commands."""

import argparse
import json
from pathlib import Path

from lfma.core import BACKBONES, GLUE_TASKS, IMAGE_TASKS
from lfma.data.manifest import read_bundle, write_bundle
from lfma.data.descriptive import dataset_statistics, label_shift
from lfma.data.duplicates import duplicate_report, deduplicate_partition
from lfma.data.sampling import stratified_subset
from lfma.adaptation.budgets import (
    support_budget,
    ratio_for_budget,
    budget_grid,
    budget_by_layer,
)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def data_audit_cli():
    parser = argparse.ArgumentParser(
        description="Inspect a portable benchmark dataset before training."
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--image-identity", choices=("bytes", "pixels", "path"), default="bytes"
    )
    parser.add_argument("--inspect-images", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    metadata, partitions = read_bundle(args.manifest_dir)
    task = metadata["task"]
    duplicates = duplicate_report(partitions, task, args.image_identity)
    report = {
        "format": "lfma-dataset-audit-v1",
        "task": task,
        "dataset": str(Path(args.manifest_dir).resolve()),
        "statistics": dataset_statistics(partitions, task, args.inspect_images),
        "label_total_variation": label_shift(partitions, task),
        "duplicates": duplicates,
    }
    save_json(args.output, report)
    print(
        json.dumps(
            {
                "task": task,
                "clean": duplicates["clean_partitions"],
                "output": args.output,
            },
            indent=2,
        )
    )
    if args.require_clean and not duplicates["clean_partitions"]:
        raise ValueError(
            "The dataset audit found shared groups or conflicting input records"
        )


def subset_bundle_cli():
    parser = argparse.ArgumentParser(
        description="Prepare a stratified training subset while retaining evaluation splits."
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-per-class", type=int, default=1)
    parser.add_argument("--deduplicate", action="store_true")
    args = parser.parse_args()
    source = Path(args.manifest_dir).resolve()
    output = Path(args.output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Write derived bundles outside the source bundle")
    metadata, partitions = read_bundle(source)
    task = metadata["task"]
    if task == "stsb":
        raise ValueError("Stratified class subsampling applies to classification tasks")
    training = partitions["train"]
    removed = []
    if args.deduplicate:
        training, removed = deduplicate_partition(training, task)
    selected = stratified_subset(
        [row["label"] for row in training],
        args.count,
        args.seed,
        args.minimum_per_class,
    )
    partitions["train"] = [training[index] for index in selected]
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    staging = output / "source-records"
    staging.mkdir(exist_ok=True)
    for split, rows in partitions.items():
        path = staging / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        sources[split] = path
    prepared = write_bundle(output, task, sources)
    save_json(
        output / "selection.json",
        {
            "source": str(source),
            "source_fingerprint": metadata["summary"]["train"]["sha256"],
            "seed": args.seed,
            "selected_ids": [row["id"] for row in partitions["train"]],
            "deduplicated": removed,
        },
    )
    print(json.dumps(prepared["summary"], indent=2))


def budget_cli():
    parser = argparse.ArgumentParser(
        description="Calculate adapter coefficients, real scalars, and storage."
    )
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--ratio", type=float)
    choice.add_argument("--real-parameters", type=int)
    parser.add_argument("--task", choices=sorted(set(GLUE_TASKS) | set(IMAGE_TASKS)))
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.task is not None:
        vision = BACKBONES[args.backbone]["family"] == "vit"
        if (args.task in IMAGE_TASKS) != vision:
            raise ValueError("Task modality must match the selected backbone")
    if args.real_parameters is not None:
        result = ratio_for_budget(args.backbone, args.real_parameters)
    else:
        classes = (
            IMAGE_TASKS.get(args.task, 1 if args.task == "stsb" else 2)
            if args.task
            else None
        )
        result = support_budget(args.backbone, args.ratio, classes)
    result["by_layer"] = budget_by_layer(result)
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, indent=2))


def budget_grid_cli():
    parser = argparse.ArgumentParser(
        description="Compare support budgets before a ratio ablation."
    )
    parser.add_argument(
        "--backbones", choices=sorted(BACKBONES), nargs="+", required=True
    )
    parser.add_argument("--ratios", type=float, nargs="+", required=True)
    parser.add_argument(
        "--tasks", nargs="+", choices=sorted(set(GLUE_TASKS) | set(IMAGE_TASKS))
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = budget_grid(args.backbones, args.ratios, args.tasks)
    if not rows:
        raise ValueError(
            "No compatible backbone-task budget combinations were selected"
        )
    save_json(args.output, {"format": "lfma-budget-grid-v1", "settings": rows})
    print(json.dumps({"settings": len(rows), "output": args.output}, indent=2))
