"""Detailed prediction analysis and paired benchmark reports."""

import argparse
import json
from pathlib import Path

from lfma.core import GLUE_TASKS
from lfma.evaluation.predictions import PredictionTable
from lfma.evaluation.metrics import (
    classification_report,
    error_records,
    most_confused_pairs,
    topk_accuracy,
)
from lfma.evaluation.metrics import (
    regression_report,
    regression_errors,
    residual_bins,
    paired_residual_change,
)
from lfma.evaluation.confidence import (
    calibration_report,
    softmax,
    selective_accuracy,
    fit_temperature,
)
from lfma.evaluation.confidence import (
    bootstrap_table,
    paired_bootstrap,
    spearman_interval,
)


def table_report(table, bootstrap=2000, seed=42, bins=15, error_limit=100):
    report = {
        "format": "lfma-prediction-analysis-v1",
        "task": table.task,
        "samples": len(table.ids),
        "fingerprint": table.fingerprint(),
    }
    if table.labels is None:
        return report
    if table.task == "stsb":
        report["regression"] = regression_report(table.predictions, table.labels)
        report["residual_bins"] = residual_bins(table.predictions, table.labels)
        report["errors"] = regression_errors(
            table.ids, table.predictions, table.labels, error_limit
        )
        if bootstrap:
            report["spearman_interval"] = spearman_interval(table, bootstrap, seed=seed)
    else:
        probabilities = softmax(table.logits)
        report["classification"] = classification_report(
            table.predictions, table.labels, table.logits.shape[1]
        )
        report["ranking"] = topk_accuracy(table.logits, table.labels)
        report["calibration"] = calibration_report(probabilities, table.labels, bins)
        report["selective_accuracy"] = selective_accuracy(probabilities, table.labels)
        report["errors"] = error_records(
            table.ids, table.predictions, table.labels, probabilities
        )[:error_limit]
        report["confused_pairs"] = most_confused_pairs(
            report["classification"]["confusion"]
        )
    if bootstrap:
        report["primary_interval"] = bootstrap_table(
            table, repeats=bootstrap, seed=seed
        )
    if table.groups is not None:
        report["groups"] = group_reports(table)
    return report


def group_reports(table):
    from collections import defaultdict

    groups = defaultdict(list)
    for index, value in enumerate(table.groups):
        groups[value].append(index)
    rows = []
    for group, indices in sorted(groups.items()):
        subset = table.subset(indices)
        if table.task == "stsb":
            metrics = regression_report(subset.predictions, subset.labels)
        else:
            metrics = classification_report(
                subset.predictions, subset.labels, subset.logits.shape[1]
            )
            metrics.pop("confusion")
            metrics.pop("classes")
        rows.append(dict(group=group, **metrics))
    return rows


def paired_report(baseline, candidate, bootstrap=2000, seed=42):
    report = paired_bootstrap(baseline, candidate, repeats=bootstrap, seed=seed)
    aligned = candidate.align_to(baseline.ids)
    if baseline.task == "stsb":
        report["residual_change"] = paired_residual_change(
            baseline.predictions, aligned.predictions, baseline.labels
        )
    else:
        before = baseline.predictions == baseline.labels
        after = aligned.predictions == baseline.labels
        report["transition_counts"] = {
            "both_correct": int((before & after).sum()),
            "both_wrong": int((~before & ~after).sum()),
            "corrected": int((~before & after).sum()),
            "regressed": int((before & ~after).sum()),
        }
        report["changed_examples"] = [
            {
                "id": identity,
                "label": int(baseline.labels[index]),
                "baseline": int(baseline.predictions[index]),
                "candidate": int(aligned.predictions[index]),
            }
            for index, identity in enumerate(baseline.ids)
            if baseline.predictions[index] != aligned.predictions[index]
        ]
    return report


def analyze_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--errors", type=int, default=100)
    args = parser.parse_args()
    if args.bootstrap < 0 or args.errors < 0:
        raise ValueError("Bootstrap repeats and error counts must be nonnegative")
    table, metadata = PredictionTable.load(args.predictions)
    report = table_report(table, args.bootstrap, args.seed, args.bins, args.errors)
    report["metadata"] = metadata
    if args.baseline:
        baseline, _ = PredictionTable.load(args.baseline)
        report["paired"] = paired_report(
            baseline, table, args.bootstrap or 2000, args.seed
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    table.export_jsonl(output / "predictions.jsonl", probabilities=True)
    print(
        json.dumps(
            {"task": table.task, "samples": len(table.ids), "output": str(output)},
            indent=2,
        )
    )


def calibrate_cli():
    parser = argparse.ArgumentParser(
        description="Fit a temperature on validation logits and apply it to a separate split."
    )
    parser.add_argument("--validation", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validation, validation_metadata = PredictionTable.load(args.validation)
    table, metadata = PredictionTable.load(args.predictions)
    if validation.task != table.task or table.task == "stsb":
        raise ValueError(
            "Temperature calibration requires the same classification task"
        )
    if validation.labels is None or validation_metadata.get("split") != "validation":
        raise ValueError("Fit temperature using labeled validation predictions")
    if metadata.get("split") not in ('validation', 'test'):
        raise ValueError("Apply calibration to a held-out prediction split")
    validation.assert_disjoint(table)
    if validation.logits.shape[1] != table.logits.shape[1]:
        raise ValueError("Class vocabularies differ between calibration and evaluation")
    result = fit_temperature(validation.logits, validation.labels)
    probabilities = softmax(table.logits, result["temperature"])
    report = {
        'fit': result, 'task': table.task, 'samples': len(table.ids),
        'validation_fingerprint': validation.fingerprint(),
        'assessment_fingerprint': table.fingerprint(),
    }
    if table.labels is not None:
        report["before"] = calibration_report(softmax(table.logits), table.labels)
        report["after"] = calibration_report(probabilities, table.labels)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    calibrated = PredictionTable(
        table.task,
        table.ids,
        table.logits / result["temperature"],
        table.labels,
        table.groups,
        table.content_keys,
    )
    calibrated.save(
        output / "calibrated.npz", dict(metadata, temperature=result["temperature"])
    )
    print(json.dumps(report, indent=2))


def primary_metric(task):
    return GLUE_TASKS.get(task, (None, None, "accuracy"))[2]
