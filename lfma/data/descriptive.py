"""Dataset measurements used before creating a benchmark training run."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from lfma.core import GLUE_TASKS, IMAGE_TASKS


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Distribution inputs must be finite scalar values")
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": len(array),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "p99": float(np.quantile(array, 0.99)),
    }


def label_statistics(rows, task):
    labels = [row["label"] for row in rows if row["label"] >= 0]
    output = {"labeled": len(labels), "unlabeled": len(rows) - len(labels)}
    if task == "stsb":
        output["scores"] = distribution(labels)
        edges = np.linspace(0, 5, 11)
        counts, _ = np.histogram(labels, bins=edges)
        output["histogram"] = [
            {"lower": float(lo), "upper": float(hi), "count": int(count)}
            for lo, hi, count in zip(edges[:-1], edges[1:], counts)
        ]
        return output
    classes = IMAGE_TASKS.get(task, 2)
    counts = Counter(labels)
    vector = np.array([counts[index] for index in range(classes)])
    present = vector[vector > 0]
    output.update(
        classes=classes,
        counts=vector.tolist(),
        missing_classes=np.flatnonzero(vector == 0).tolist(),
        largest_smallest_ratio=float(present.max() / present.min())
        if len(present)
        else None,
    )
    return output


def text_statistics(rows, task):
    first, second, _ = GLUE_TASKS[task]
    report = {}
    for field in (first, second):
        if field is None:
            continue
        values = [row[field] for row in rows]
        report[field] = {
            "characters": distribution([len(value) for value in values]),
            "whitespace_tokens": distribution([len(value.split()) for value in values]),
            "unique": len(set(values)),
            "empty": sum(not value.strip() for value in values),
        }
    if second:
        report["identical_pairs"] = sum(row[first] == row[second] for row in rows)
    return report


def image_statistics(rows):
    from PIL import Image

    widths, heights, ratios, modes = [], [], [], Counter()
    failures = []
    for row in rows:
        try:
            with Image.open(row["image"]) as image:
                image.verify()
            with Image.open(row["image"]) as image:
                width, height = image.size
                if min(width, height) < 1:
                    raise ValueError("Empty image dimensions")
                widths.append(width)
                heights.append(height)
                ratios.append(width / height)
                modes[image.mode] += 1
        except (OSError, ValueError) as error:
            failures.append({"id": row["id"], "error": str(error)})
    return {
        "width": distribution(widths),
        "height": distribution(heights),
        "aspect_ratio": distribution(ratios),
        "modes": dict(modes),
        "unreadable": failures,
    }


def dataset_statistics(partitions, task, inspect_images=False):
    result = {}
    for split, rows in partitions.items():
        report = {
            "records": len(rows),
            "labels": label_statistics(rows, task),
            "groups": len({row["group"] for row in rows if row.get("group")}),
        }
        if task in GLUE_TASKS:
            report["text"] = text_statistics(rows, task)
        elif inspect_images:
            report["images"] = image_statistics(rows)
        result[split] = report
    return result


def label_shift(partitions, task):
    if task == "stsb":
        bins = np.linspace(0, 5, 11)
        histograms = {
            split: np.histogram(
                [row["label"] for row in rows if row["label"] >= 0], bins
            )[0]
            for split, rows in partitions.items()
        }
    else:
        classes = IMAGE_TASKS.get(task, 2)
        histograms = {
            split: np.bincount(
                [row["label"] for row in rows if row["label"] >= 0], minlength=classes
            )
            for split, rows in partitions.items()
        }
    normalized = {
        key: values / values.sum()
        for key, values in histograms.items()
        if values.sum() > 0
    }
    return {
        left: {
            right: float(np.abs(normalized[left] - normalized[right]).sum() / 2)
            for right in normalized
        }
        for left in normalized
    }


def _image_key(image):
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return _image_key(source)
    if isinstance(image, dict):
        if image.get('bytes') is not None:
            from io import BytesIO

            with Image.open(BytesIO(image['bytes'])) as source:
                return _image_key(source)
        return _image_key(image['path'])
    if not hasattr(image, 'convert'):
        image = Image.fromarray(np.asarray(image))
    image = image.convert('RGB')
    digest = hashlib.sha256()
    digest.update(f'RGB:{image.width}:{image.height}:'.encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _source_record(dataset, index):
    from torch.utils.data import Subset

    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    records = getattr(dataset, 'records', None)
    row = records[index] if records is not None else dataset[index]
    return row, index


def dataset_identity(dataset, task):
    if len(dataset) < 1:
        raise ValueError('A dataset identity requires at least one example')
    identities, groups, keys, labels = [], [], [], []
    for index in range(len(dataset)):
        record, source_index = _source_record(dataset, index)
        if task in GLUE_TASKS:
            first, second, _ = GLUE_TASKS[task]
            content = {field: record[field] for field in (first, second) if field}
            serialized = json.dumps(content, sort_keys=True, ensure_ascii=False)
            key = hashlib.sha256(serialized.encode()).hexdigest()
            label = record.get('label', -1)
        else:
            image, label = (record['image'], record['label']) if isinstance(
                record, dict
            ) else record
            key = _image_key(image)
        identity = record.get('id') if isinstance(record, dict) else None
        group = record.get('group') if isinstance(record, dict) else None
        identities.append(str(identity) if identity is not None else
                          f'{source_index}:{key[:16]}')
        groups.append(str(group) if group is not None else None)
        keys.append(key)
        labels.append(float(label) if label is not None else -1.0)
    if len(set(identities)) != len(identities):
        raise ValueError('Dataset identities must be unique within a partition')
    if not np.isfinite(labels).all():
        raise ValueError('Dataset labels must be finite before training')
    if any(group is not None for group in groups) and any(
        group is None for group in groups
    ):
        raise ValueError('Group membership must be supplied for every example or none')
    groups = groups if groups[0] is not None else None
    payload = dict(task=task, ids=identities, groups=groups, content=keys, labels=labels)
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode()).hexdigest()
    return {
        'format': 'lfma-dataset-identity-v1',
        'task': task,
        'samples': len(dataset),
        'sha256': digest,
        'ids': identities,
        'groups': groups,
        'content_keys': keys,
    }


def loader_identity(loader, task):
    identity = getattr(loader, 'lfma_identity', None)
    if identity is None:
        identity = dataset_identity(loader.dataset, task)
        loader.lfma_identity = identity
    if identity['samples'] != len(loader.dataset) or identity['task'] != task:
        raise ValueError('Cached dataset identity no longer matches this loader')
    return identity


def training_data_contract(train_loader, validation_loader, task):
    contract = {'format': 'lfma-training-data-v1', 'task': task}
    for name, loader in (('train', train_loader), ('validation', validation_loader)):
        identity = loader_identity(loader, task)
        contract[name] = {key: identity[key] for key in ('samples', 'sha256')}
        contract[name]['batch_size'] = loader.batch_size
        contract[name]['batches'] = len(loader)
        contract[name]['drop_last'] = loader.drop_last
    return contract
