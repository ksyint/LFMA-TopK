"""Stable prediction tables shared by inference and detailed task reports."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


class PredictionTable:
    def __init__(self, task, ids, logits, labels=None, groups=None, content_keys=None):
        self.task = str(task)
        self.ids = [str(value) for value in ids]
        self.logits = np.asarray(logits, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels)
        self.groups = None if groups is None else [str(value) for value in groups]
        self.content_keys = None if content_keys is None else list(content_keys)
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Prediction identities must be nonempty and unique")
        if self.logits.ndim != 2 or self.logits.shape[0] != len(self.ids):
            raise ValueError("Logits must have one matrix row per identity")
        if not np.isfinite(self.logits).all():
            raise ValueError("Saved logits must be finite")
        if self.labels is not None:
            if (
                self.labels.shape != (len(self.ids),)
                or not np.isfinite(self.labels).all()
            ):
                raise ValueError(
                    "References must contain one finite value per identity"
                )
        if self.groups is not None and len(self.groups) != len(self.ids):
            raise ValueError("Group identities must align with predictions")
        if self.task == "stsb" and self.logits.shape[1] != 1:
            raise ValueError("STS-B requires one regression score per row")
        if self.task != "stsb" and self.logits.shape[1] < 2:
            raise ValueError("Classification requires at least two score columns")
        if self.labels is not None:
            if self.task == 'stsb':
                if np.any((self.labels < 0) | (self.labels > 5)):
                    raise ValueError('STS-B references must lie in [0,5]')
                self.labels = self.labels.astype(np.float32)
            else:
                if not np.equal(self.labels, np.floor(self.labels)).all():
                    raise ValueError('Classification references must be integer IDs')
                if np.any((self.labels < 0) | (self.labels >= self.logits.shape[1])):
                    raise ValueError('Classification references exceed the vocabulary')
                self.labels = self.labels.astype(np.int64)
        if self.content_keys is not None:
            if len(self.content_keys) != len(self.ids):
                raise ValueError('Content fingerprints must align with predictions')
            if any(not isinstance(key, str) or len(key) != 64 or
                   any(letter not in '0123456789abcdef' for letter in key)
                   for key in self.content_keys):
                raise ValueError('Content fingerprints must be SHA-256 hex strings')

    @property
    def predictions(self):
        return self.logits[:, 0] if self.task == "stsb" else self.logits.argmax(1)

    def fingerprint(self, include_identity=True):
        hasher = hashlib.sha256()
        hasher.update(self.task.encode())
        hasher.update(json.dumps(self.ids).encode())
        hasher.update(self.logits.tobytes())
        if self.labels is not None:
            hasher.update(self.labels.tobytes())
        if include_identity:
            hasher.update(json.dumps(self.groups).encode())
            hasher.update(json.dumps(self.content_keys).encode())
        return hasher.hexdigest()

    def subset(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("Prediction subsets require a nonempty index vector")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('Prediction subset indices must be integers')
        if np.any((indices < 0) | (indices >= len(self.ids))):
            raise ValueError('Prediction subset indices exceed the table bounds')
        return PredictionTable(
            self.task,
            [self.ids[index] for index in indices],
            self.logits[indices],
            None if self.labels is None else self.labels[indices],
            None if self.groups is None else [self.groups[index] for index in indices],
            None if self.content_keys is None else
            [self.content_keys[index] for index in indices],
        )

    def align_to(self, ids):
        indices = {identity: index for index, identity in enumerate(self.ids)}
        if set(ids) != set(self.ids) or len(ids) != len(self.ids):
            raise ValueError(
                "Paired prediction tables must contain the same identities"
            )
        return self.subset([indices[identity] for identity in ids])

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        description = dict(metadata or {})
        description.update(
            format="lfma-predictions-v2",
            task=self.task,
            labeled=self.labels is not None,
            grouped=self.groups is not None,
            fingerprint=self.fingerprint(),
        )
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(
                    stream,
                    ids=np.asarray(self.ids),
                    logits=self.logits,
                    labels=self.labels if self.labels is not None else np.empty(0),
                    groups=np.asarray(self.groups or [], dtype=str),
                    content_keys=np.asarray(self.content_keys or [], dtype=str),
                    metadata=np.asarray(json.dumps(description, allow_nan=False)),
                )
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            version = metadata.get('format')
            if version not in ('lfma-predictions-v1', 'lfma-predictions-v2'):
                raise ValueError("Unsupported prediction archive")
            table = cls(
                metadata["task"],
                archive["ids"].tolist(),
                archive["logits"],
                archive["labels"] if metadata["labeled"] else None,
                archive["groups"].tolist() if metadata["grouped"] else None,
                archive['content_keys'].tolist()
                if 'content_keys' in archive and len(archive['content_keys']) else None,
            )
            if version == 'lfma-predictions-v1':
                legacy_labels = archive['labels'] if metadata['labeled'] else None
                normalized_labels = table.labels
                table.labels = legacy_labels
                fingerprint = table.fingerprint(include_identity=False)
                table.labels = normalized_labels
            else:
                fingerprint = table.fingerprint()
        if fingerprint != metadata["fingerprint"]:
            raise ValueError("Prediction content differs from its fingerprint")
        return table, metadata

    def export_jsonl(self, path, probabilities=False):
        from lfma.evaluation.confidence import softmax

        scores = softmax(self.logits) if probabilities and self.task != "stsb" else None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for index, identity in enumerate(self.ids):
                row = {"id": identity, "prediction": self.predictions[index].item()}
                if self.labels is not None:
                    row["label"] = self.labels[index].item()
                if self.groups is not None:
                    row["group"] = self.groups[index]
                if scores is not None:
                    row["probabilities"] = scores[index].tolist()
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )

    def assert_disjoint(self, other):
        if self.content_keys is not None and other.content_keys is not None:
            overlap = set(self.content_keys) & set(other.content_keys)
            kind = 'input content'
        else:
            overlap = set(self.ids) & set(other.ids)
            kind = 'example identities'
        if overlap:
            raise ValueError(f'Calibration partitions share {len(overlap)} {kind}')
        if self.groups is not None and other.groups is not None:
            overlap = set(self.groups) & set(other.groups)
            if overlap:
                raise ValueError('Calibration partitions share example groups')

