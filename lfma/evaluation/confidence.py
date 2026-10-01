"""Probability calibration measured on held-out classification predictions."""

from collections import defaultdict

import numpy as np

from lfma.evaluation.metrics import correlation, rank_values


def softmax(logits, temperature=1.0):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] < 2:
        raise ValueError("Logits must have shape examples by classes")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(
            "Logits and temperature must be finite with positive temperature"
        )
    values = values / temperature
    values -= values.max(1, keepdims=True)
    exponent = np.exp(values)
    return exponent / exponent.sum(1, keepdims=True)


def validate_probabilities(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if probabilities.ndim != 2 or labels.shape != (len(probabilities),):
        raise ValueError("Probability rows and labels must be aligned")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(1), 1, atol=1e-6):
        raise ValueError("Each probability row must sum to one")
    if not np.equal(labels, labels.astype(np.int64)).all():
        raise ValueError("Calibration labels must be integers")
    labels = labels.astype(np.int64)
    if np.any((labels < 0) | (labels >= probabilities.shape[1])):
        raise ValueError("Calibration labels exceed class count")
    return probabilities, labels


def reliability_diagram(probabilities, labels, bins=15):
    probabilities, labels = validate_probabilities(probabilities, labels)
    if bins < 1 or len(labels) == 0:
        raise ValueError("Reliability requires examples and positive bin count")
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    records = []
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        average = float(confidence[selected].mean()) if count else None
        records.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "accuracy": accuracy,
                "confidence": average,
                "gap": abs(accuracy - average) if count else None,
            }
        )
    return records


def calibration_report(probabilities, labels, bins=15):
    probabilities, labels = validate_probabilities(probabilities, labels)
    records = reliability_diagram(probabilities, labels, bins)
    target = probabilities[np.arange(len(labels)), labels]
    squared = np.square(probabilities).sum(1) - 2 * target + 1
    return {
        "samples": len(labels),
        "negative_log_likelihood": float(
            -np.log(np.maximum(target, np.finfo(float).tiny)).mean()
        ),
        "brier": float(squared.mean()),
        "ece": sum(row["count"] * (row["gap"] or 0) for row in records) / len(labels),
        "maximum_gap": max((row["gap"] for row in records if row["count"]), default=0),
        "reliability": records,
    }


def fit_temperature(logits, labels, lower=0.05, upper=20.0, iterations=80):
    logits = np.asarray(logits, dtype=np.float64)
    _, labels = validate_probabilities(softmax(logits), labels)
    if not 0 < lower < upper or iterations < 1:
        raise ValueError("Invalid temperature search bounds")

    def objective(log_temperature):
        values = logits / np.exp(log_temperature)
        maximum = values.max(1)
        normalizer = maximum + np.log(np.exp(values - maximum[:, None]).sum(1))
        return float((normalizer - values[np.arange(len(labels)), labels]).mean())

    lo, hi = np.log(lower), np.log(upper)
    ratio = (np.sqrt(5) - 1) / 2
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    for _ in range(iterations):
        if objective(left) < objective(right):
            hi, right = right, left
            left = hi - ratio * (hi - lo)
        else:
            lo, left = left, right
            right = lo + ratio * (hi - lo)
    candidates = [np.log(lower), np.log(upper), (lo + hi) / 2]
    if lower <= 1 <= upper:
        candidates.append(0.0)
    selected = min(candidates, key=objective)
    return {
        "temperature": float(np.exp(selected)),
        "nll": objective(selected),
        "samples": len(labels),
    }


def selective_accuracy(probabilities, labels, coverages=(1.0, 0.9, 0.75, 0.5)):
    probabilities, labels = validate_probabilities(probabilities, labels)
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    order = np.argsort(-confidence, kind="stable")
    rows = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("Coverage must lie in (0,1]")
        count = max(1, int(np.ceil(coverage * len(labels))))
        threshold = confidence[order[count - 1]]
        chosen = np.flatnonzero(confidence >= threshold)
        count = len(chosen)
        rows.append(
            {
                "requested_coverage": coverage,
                "coverage": count / len(labels),
                "samples": count,
                "accuracy": float(correct[chosen].mean()),
                "minimum_confidence": float(confidence[chosen].min()),
            }
        )
    return rows


def primary_score(task, predictions, labels):
    if task == "stsb":
        return correlation(predictions.astype(float), labels.astype(float))
    if task == "cola":
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float((tp * tn - fp * fn) / denominator) if denominator else 0.0
    return float((predictions == labels).mean())


def resampling_units(count, groups=None):
    if count < 1:
        raise ValueError("Bootstrap requires examples")
    if groups is None:
        return [np.asarray([index]) for index in range(count)]
    if len(groups) != count:
        raise ValueError("Group assignments must align with predictions")
    units = defaultdict(list)
    for index, group in enumerate(groups):
        units[str(group)].append(index)
    return [np.asarray(indices) for _, indices in sorted(units.items())]


def percentile_interval(values, confidence):
    values = np.asarray(values, dtype=np.float64)
    if not 0 < confidence < 1 or not np.isfinite(values).all() or len(values) < 1:
        raise ValueError("Invalid bootstrap values or confidence")
    tail = (1 - confidence) / 2
    return {
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1 - tail)),
        "standard_error": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "confidence": confidence,
    }


def bootstrap_table(table, repeats=2000, confidence=0.95, seed=42):
    if table.labels is None or repeats < 1:
        raise ValueError("Bootstrap requires labeled predictions and positive repeats")
    units = resampling_units(len(table.ids), table.groups)
    rng = np.random.default_rng(seed)
    predictions = table.predictions
    values = np.empty(repeats)
    for repeat in range(repeats):
        selected = rng.integers(0, len(units), size=len(units))
        indices = np.concatenate([units[index] for index in selected])
        values[repeat] = primary_score(
            table.task, predictions[indices], table.labels[indices]
        )
    return {
        "estimate": primary_score(table.task, predictions, table.labels),
        "samples": len(table.ids),
        "units": len(units),
        "resampling": "groups" if table.groups is not None else "examples",
        "repeats": repeats,
        "seed": seed,
        **percentile_interval(values, confidence),
    }


def paired_bootstrap(baseline, candidate, repeats=2000, confidence=0.95, seed=42):
    candidate = candidate.align_to(baseline.ids)
    if baseline.task != candidate.task:
        raise ValueError("Paired task names must match")
    if baseline.labels is None or candidate.labels is None:
        raise ValueError("Paired comparisons require labeled tables")
    if not np.array_equal(baseline.labels, candidate.labels):
        raise ValueError("Paired tables disagree on reference labels")
    if baseline.groups != candidate.groups:
        raise ValueError("Paired tables disagree on bootstrap group membership")
    if (baseline.content_keys is not None and candidate.content_keys is not None
            and baseline.content_keys != candidate.content_keys):
        raise ValueError('Paired identities refer to different input content')
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    units = resampling_units(len(baseline.ids), baseline.groups)
    before, after, labels = baseline.predictions, candidate.predictions, baseline.labels
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats)
    for repeat in range(repeats):
        selected = rng.integers(0, len(units), size=len(units))
        indices = np.concatenate([units[index] for index in selected])
        differences[repeat] = primary_score(
            baseline.task, after[indices], labels[indices]
        ) - primary_score(baseline.task, before[indices], labels[indices])
    return {
        "baseline": primary_score(baseline.task, before, labels),
        "candidate": primary_score(baseline.task, after, labels),
        "difference": primary_score(baseline.task, after, labels)
        - primary_score(baseline.task, before, labels),
        "samples": len(labels),
        "units": len(units),
        "repeats": repeats,
        **percentile_interval(differences, confidence),
    }


def spearman_interval(table, repeats=2000, confidence=0.95, seed=42):
    if table.task != "stsb" or table.labels is None:
        raise ValueError("Rank correlation intervals require labeled STS-B predictions")
    units = resampling_units(len(table.ids), table.groups)
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = np.concatenate(
            [units[index] for index in generator.integers(0, len(units), len(units))]
        )
        values.append(
            correlation(
                rank_values(table.predictions[indices]),
                rank_values(table.labels[indices]),
            )
        )
    return {
        "estimate": correlation(
            rank_values(table.predictions), rank_values(table.labels)
        ),
        **percentile_interval(values, confidence),
    }
