"""Append-only epoch records with explicit resume boundaries."""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch


def configuration_digest(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class RunJournal:
    def __init__(self, directory, config, resume_epoch=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "epochs.jsonl"
        self.config_path = self.directory / "run.json"
        self.digest = configuration_digest(config)
        self.rows = []
        if resume_epoch is None:
            if self.path.exists() or self.config_path.exists():
                raise FileExistsError(
                    "Select a new output directory or resume its saved checkpoint"
                )
            self.config_path.write_text(
                json.dumps(
                    {
                        "format": "lfma-training-run-v1",
                        "config": config,
                        "sha256": self.digest,
                        "created": time.time(),
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
        else:
            self._restore(resume_epoch)

    def _restore(self, epoch):
        metadata = json.loads(self.config_path.read_text())
        if metadata["sha256"] != self.digest:
            raise ValueError("The journal configuration changed during resume")
        lines = self.path.read_text().splitlines() if self.path.exists() else []
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and len(self.rows) == epoch:
                    break
                raise ValueError('Epoch journal contains an incomplete committed row')
            if row["epoch"] <= epoch:
                self.rows.append(row)
        expected = list(range(1, epoch + 1))
        if [row["epoch"] for row in self.rows] != expected:
            raise ValueError("Epoch journal does not cover the resumed checkpoint")
        temporary = self.path.with_suffix(".resume.tmp")
        temporary.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in self.rows)
        )
        temporary.replace(self.path)

    def append(self, row):
        expected = self.rows[-1]["epoch"] + 1 if self.rows else 1
        if row["epoch"] != expected:
            raise ValueError("Epoch records must be appended in consecutive order")
        payload = json.dumps(row, allow_nan=False) + "\n"
        with self.path.open("a") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(dict(row))

    def best(self, metric):
        if not self.rows:
            return None
        return max(self.rows, key=lambda row: row["validation"][metric])

    def summary(self, metric):
        best = self.best(metric)
        return {
            "epochs": len(self.rows),
            "metric": metric,
            "best_epoch": best["epoch"] if best else None,
            "best_score": best["validation"][metric] if best else None,
            "wall_seconds": sum(row.get("wall_seconds", 0) for row in self.rows),
            "config_sha256": self.digest,
        }


class EarlyStopping:
    def __init__(self, patience=0, minimum_improvement=0.0):
        if isinstance(patience, bool) or int(patience) != patience or patience < 0:
            raise ValueError('Stopping patience must be a nonnegative integer')
        if not math.isfinite(minimum_improvement) or minimum_improvement < 0:
            raise ValueError('Minimum improvement must be finite and nonnegative')
        self.patience = int(patience)
        self.minimum_improvement = float(minimum_improvement)
        self.best = None
        self.anchor = None
        self.stale = 0
        self.epochs = 0

    def update(self, score):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError('Checkpoint selection requires a finite validation score')
        improved = self.best is None or score > self.best
        if self.anchor is None or score > self.anchor + self.minimum_improvement:
            self.anchor = score
            self.stale = 0
        else:
            self.stale += 1
        if improved:
            self.best = score
        self.epochs += 1
        return improved

    @property
    def stopped(self):
        return self.patience > 0 and self.stale >= self.patience

    def state_dict(self):
        return {
            'patience': self.patience,
            'minimum_improvement': self.minimum_improvement,
            'best': self.best,
            'anchor': self.anchor,
            'stale': self.stale,
            'epochs': self.epochs,
        }

    def restore(self, rows, metric, saved=None):
        if self.epochs:
            raise RuntimeError('Stopping history can only be restored once')
        for epoch, row in enumerate(rows, start=1):
            if row['epoch'] != epoch:
                raise ValueError('Stopping history must have consecutive epochs')
            self.update(row['validation'][metric])
        if saved is not None and saved != self.state_dict():
            raise ValueError('Checkpoint stopping state differs from its epoch journal')


class EpochTimer:
    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Training timers require CUDA")
        self.started = None
        self.samples = 0
        self.steps = 0

    def __enter__(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        return self

    def update(self, samples):
        self.samples += int(samples)
        self.steps += 1

    def __exit__(self, kind, value, traceback):
        torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.started
        self.peak_allocated = torch.cuda.max_memory_allocated(self.device)
        self.peak_reserved = torch.cuda.max_memory_reserved(self.device)

    def record(self):
        if not hasattr(self, "elapsed"):
            raise RuntimeError("Timing interval has not completed")
        return {
            "seconds": self.elapsed,
            "samples": self.samples,
            "steps": self.steps,
            "samples_per_second": self.samples / max(self.elapsed, 1e-9),
            "peak_allocated_bytes": self.peak_allocated,
            "peak_reserved_bytes": self.peak_reserved,
        }
