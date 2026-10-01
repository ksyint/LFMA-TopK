"""Classification scores, STS-B residuals, and aligned error records."""

import numpy as np


def validated_labels(predictions, labels, classes):
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.ndim != 1 or predictions.shape != labels.shape:
        raise ValueError("Predictions and references must be aligned vectors")
    if classes < 2 or len(labels) == 0:
        raise ValueError("Classification requires examples and at least two classes")
    for name, array in (("prediction", predictions), ("reference", labels)):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError(f"{name} IDs must be finite integers")
        if np.any((array < 0) | (array >= classes)):
            raise ValueError(f"{name} IDs exceed the class vocabulary")
    return predictions.astype(np.int64), labels.astype(np.int64)


def confusion_matrix(predictions, labels, classes):
    predictions, labels = validated_labels(predictions, labels, classes)
    flat = np.bincount(classes * labels + predictions, minlength=classes * classes)
    return flat.reshape(classes, classes)


def divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(np.asarray(numerator), dtype=np.float64),
        where=np.asarray(denominator) != 0,
    )


def classification_report(predictions, labels, classes, names=None):
    matrix = confusion_matrix(predictions, labels, classes)
    names = list(map(str, range(classes))) if names is None else list(names)
    if len(names) != classes or len(set(names)) != classes:
        raise ValueError("Class names must be distinct and cover the vocabulary")
    correct = np.diag(matrix)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    precision = divide(correct, predicted)
    recall = divide(correct, support)
    f1 = divide(2 * precision * recall, precision + recall)
    present = support > 0
    per_class = [
        {
            "index": index,
            "name": names[index],
            "support": int(support[index]),
            "predicted": int(predicted[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index in range(classes)
    ]
    return {
        "samples": int(matrix.sum()),
        "accuracy": float(correct.sum() / matrix.sum()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / support.sum()),
        "classes": per_class,
        "confusion": matrix.tolist(),
    }


def topk_accuracy(logits, labels, ks=(1, 5)):
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Ranking scores must form a finite matrix")
    labels = np.asarray(labels)
    validated_labels(np.zeros(len(labels)), labels, scores.shape[1])
    if len(scores) != len(labels):
        raise ValueError("Scores and references must have equal row counts")
    order = np.argsort(-scores, axis=1, kind="stable")
    result = {}
    for k in ks:
        if k < 1:
            raise ValueError("Top-k cutoffs must be positive")
        cutoff = min(int(k), scores.shape[1])
        result[f"top{cutoff}"] = float(
            np.any(order[:, :cutoff] == labels[:, None], axis=1).mean()
        )
    return result


def error_records(ids, predictions, labels, probabilities=None):
    if len(ids) != len(predictions) or len(labels) != len(ids):
        raise ValueError("IDs, predictions, and references must be aligned")
    if len(set(ids)) != len(ids):
        raise ValueError("Prediction IDs must be unique")
    rows = []
    for index, (identity, prediction, label) in enumerate(
        zip(ids, predictions, labels)
    ):
        if prediction == label:
            continue
        row = {"id": identity, "prediction": int(prediction), "label": int(label)}
        if probabilities is not None:
            row["confidence"] = float(probabilities[index, prediction])
            row["reference_probability"] = float(probabilities[index, label])
            row["margin"] = row["confidence"] - row["reference_probability"]
        rows.append(row)
    return sorted(rows, key=lambda row: -row.get("confidence", 0))


def most_confused_pairs(matrix, count=20):
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Confusion matrix must be square")
    pairs = [
        {"reference": int(i), "prediction": int(j), "count": int(values[i, j])}
        for i, j in zip(*np.nonzero(values))
        if i != j
    ]
    return sorted(
        pairs, key=lambda row: (-row["count"], row["reference"], row["prediction"])
    )[:count]


def paired_values(predictions, labels):
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if predictions.ndim != 1 or predictions.shape != labels.shape or len(labels) == 0:
        raise ValueError("Regression requires nonempty aligned vectors")
    if not np.isfinite(predictions).all() or not np.isfinite(labels).all():
        raise ValueError("Regression values must be finite")
    return predictions, labels


def rank_values(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def correlation(left, right):
    x = left - left.mean()
    y = right - right.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else 0.0


def regression_report(predictions, labels):
    predictions, labels = paired_values(predictions, labels)
    residual = predictions - labels
    error = np.abs(residual)
    variance = np.square(labels - labels.mean()).sum()
    return {
        "samples": len(labels),
        "pearson": correlation(predictions, labels),
        "spearman": correlation(rank_values(predictions), rank_values(labels)),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "median_absolute_error": float(np.median(error)),
        "bias": float(residual.mean()),
        "residual_std": float(residual.std()),
        "r2": float(1 - np.square(residual).sum() / variance) if variance > 0 else None,
        "prediction_minimum": float(predictions.min()),
        "prediction_maximum": float(predictions.max()),
        "outside_score_range": int(np.sum((predictions < 0) | (predictions > 5))),
    }


def residual_bins(predictions, labels, edges=None):
    predictions, labels = paired_values(predictions, labels)
    edges = np.asarray(edges if edges is not None else np.linspace(0, 5, 6))
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("Residual bins need strictly increasing edges")
    if labels.min() < edges[0] or labels.max() > edges[-1]:
        raise ValueError("Residual bins must cover every reference score")
    bins = np.minimum(np.searchsorted(edges, labels, side="right") - 1, len(edges) - 2)
    result = []
    for index in range(len(edges) - 1):
        selected = bins == index
        values = {
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(selected.sum()),
        }
        if selected.any():
            residual = predictions[selected] - labels[selected]
            values.update(
                mean_prediction=float(predictions[selected].mean()),
                mean_reference=float(labels[selected].mean()),
                bias=float(residual.mean()),
                mae=float(np.abs(residual).mean()),
            )
        result.append(values)
    return result


def regression_errors(ids, predictions, labels, count=100):
    predictions, labels = paired_values(predictions, labels)
    if len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("Regression IDs must be unique and aligned")
    error = np.abs(predictions - labels)
    order = np.argsort(-error, kind="stable")[:count]
    return [
        {
            "id": ids[index],
            "prediction": float(predictions[index]),
            "label": float(labels[index]),
            "absolute_error": float(error[index]),
        }
        for index in order
    ]


def paired_residual_change(before, after, labels):
    before, labels = paired_values(before, labels)
    after, _ = paired_values(after, labels)
    old_error = np.abs(before - labels)
    new_error = np.abs(after - labels)
    difference = new_error - old_error
    return {
        "samples": len(labels),
        "improved": int(np.sum(difference < 0)),
        "unchanged": int(np.sum(difference == 0)),
        "worsened": int(np.sum(difference > 0)),
        "mae_change": float(difference.mean()),
        "rmse_change": float(
            np.sqrt(np.square(after - labels).mean())
            - np.sqrt(np.square(before - labels).mean())
        ),
    }
