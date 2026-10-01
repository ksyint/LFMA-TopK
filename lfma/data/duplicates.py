"""Content identities across train, validation, and test partitions."""

import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from lfma.core import GLUE_TASKS


def normalized_text(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def file_digest(path, block_size=1024 * 1024):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def input_identity(row, task, image_mode="bytes"):
    if task in GLUE_TASKS:
        first, second, _ = GLUE_TASKS[task]
        values = [normalized_text(row[first])]
        if second:
            values.append(normalized_text(row[second]))
        payload = json.dumps(values, ensure_ascii=False).encode()
        return hashlib.sha256(payload).hexdigest()
    path = Path(row["image"])
    if image_mode == "bytes":
        return file_digest(path)
    if image_mode == "pixels":
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            hasher = hashlib.sha256(str(image.size).encode())
            hasher.update(image.tobytes())
            return hasher.hexdigest()
    if image_mode == "path":
        return hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    raise ValueError("Image identity mode must be bytes, pixels, or path")


def duplicate_groups(partitions, task, image_mode="bytes"):
    groups = defaultdict(list)
    for split, rows in partitions.items():
        for row in rows:
            groups[input_identity(row, task, image_mode)].append(
                {"split": split, "id": row["id"], "label": row["label"]}
            )
    duplicates = []
    for key, entries in groups.items():
        if len(entries) < 2:
            continue
        splits = {row["split"] for row in entries}
        labels = {row["label"] for row in entries if row["label"] != -1}
        duplicates.append(
            {
                "sha256": key,
                "entries": entries,
                "cross_split": len(splits) > 1,
                "label_conflict": len(labels) > 1,
            }
        )
    return sorted(duplicates, key=lambda row: row["sha256"])


def overlap_matrix(partitions, task, image_mode="path"):
    keys = {
        split: {input_identity(row, task, image_mode) for row in rows}
        for split, rows in partitions.items()
    }
    return {
        left: {right: len(keys[left] & keys[right]) for right in keys} for left in keys
    }


def group_overlap(partitions):
    locations = defaultdict(set)
    for split, rows in partitions.items():
        for row in rows:
            if row.get("group"):
                locations[str(row["group"])].add(split)
    return [
        {"group": group, "splits": sorted(splits)}
        for group, splits in sorted(locations.items())
        if len(splits) > 1
    ]


def deduplicate_partition(rows, task, image_mode="bytes"):
    seen = {}
    retained, removed = [], []
    for row in rows:
        key = input_identity(row, task, image_mode)
        if key in seen:
            previous = seen[key]
            if row["label"] != previous["label"]:
                raise ValueError(
                    f"Cannot discard conflicting labels: {previous['id']}, {row['id']}"
                )
            removed.append({"id": row["id"], "retained_id": previous["id"]})
        else:
            seen[key] = row
            retained.append(row)
    return retained, removed


def duplicate_report(partitions, task, image_mode="bytes"):
    duplicates = duplicate_groups(partitions, task, image_mode)
    groups = group_overlap(partitions)
    return {
        "identity": "normalized-text" if task in GLUE_TASKS else image_mode,
        "duplicate_groups": duplicates,
        "cross_split_groups": sum(row["cross_split"] for row in duplicates),
        "conflicting_groups": sum(row["label_conflict"] for row in duplicates),
        "shared_groups": groups,
        "clean_partitions": not groups
        and not any(row["cross_split"] or row["label_conflict"] for row in duplicates),
    }
