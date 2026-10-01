"""Supervised CUDA epochs with detailed predictions and finite-gradient checks."""

from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F

from lfma.data.streams import task_metrics
from lfma.evaluation.predictions import PredictionTable
from lfma.data.descriptive import loader_identity
from lfma.training.journal import EpochTimer
from lfma.training.precision import (
    PrecisionPolicy,
    accumulation_window,
    assert_cuda_model,
    move_batch,
)


class EpochEngine:
    def __init__(self, model, config, optimizer=None, schedule=None):
        self.model = model
        self.config = config
        self.device = assert_cuda_model(model)
        self.optimizer = optimizer
        self.schedule = schedule
        self.precision = PrecisionPolicy(
            config["train"]["precision"],
            config["train"].get("max_grad_norm", 1.0),
        )
        self.task = config["data"]["task"]
        self.accumulation = config["train"]["gradient_accumulation"]

    def forward(self, batch):
        if isinstance(batch, Mapping):
            labels = batch.get("labels")
            result = self.model(**batch)
            return result.logits, result.loss, labels
        images, labels = batch
        logits = self.model(images)
        if not isinstance(logits, torch.Tensor):
            logits = logits.logits
        loss = F.cross_entropy(logits, labels)
        return logits, loss, labels

    def run(self, loader, epoch=0, training=False, collect=False):
        if len(loader) == 0:
            raise ValueError("Cannot run an empty epoch")
        if training and self.optimizer is None:
            raise ValueError("Training requires an optimizer")
        if training and collect:
            raise ValueError("Prediction archives use ordered evaluation loaders")
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        if collect:
            if loader.drop_last or list(loader.sampler) != list(range(len(loader.dataset))):
                raise ValueError('Prediction collection requires a complete ordered sampler')
        self.model.train(training)
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        predictions, references, logits_list = [], [], []
        weighted_loss, labeled_count, skipped_steps = 0.0, 0, 0
        gradient_norms = []
        window_examples = 0
        identity = loader_identity(loader, self.task) if collect else None
        with EpochTimer(self.device) as timer:
            for step, item in enumerate(loader):
                batch = move_batch(item, self.device)
                with torch.set_grad_enabled(training), self.precision.context():
                    logits, loss, labels = self.forward(batch)
                if not torch.isfinite(logits.detach()).all():
                    raise RuntimeError(
                        f"Nonfinite logits at epoch {epoch}, batch {step}"
                    )
                if training:
                    if loss is None:
                        raise ValueError("Training examples must contain labels")
                    _, boundary = accumulation_window(
                        step, len(loader), self.accumulation
                    )
                    examples = len(labels)
                    self.precision.backward(loss * examples)
                    window_examples += examples
                    if boundary:
                        update = self.precision.step(
                            self.optimizer, self.model.parameters(), window_examples
                        )
                        window_examples = 0
                        skipped_steps += int(update["skipped"])
                        if not update["skipped"]:
                            gradient_norms.append(update["gradient_norm"])
                            if self.schedule is not None:
                                self.schedule.step()
                raw = logits.detach().float().cpu().numpy()
                estimate = raw.reshape(-1) if self.task == "stsb" else raw.argmax(-1)
                predictions.extend(estimate.tolist())
                if collect:
                    logits_list.append(raw.reshape(len(raw), -1))
                if labels is not None:
                    if loss is None or not torch.isfinite(loss.detach()):
                        raise RuntimeError(
                            "Labeled evaluation produced a nonfinite loss"
                        )
                    references.extend(labels.detach().cpu().tolist())
                    weighted_loss += float(loss.detach()) * len(labels)
                    labeled_count += len(labels)
                timer.update(len(raw))
        result = {"samples": len(predictions), "timing": timer.record()}
        if references:
            result.update(task_metrics(self.task, predictions, references))
            result["loss"] = weighted_loss / labeled_count
        if training:
            result["updates"] = self.precision.updates
            result["skipped_updates"] = skipped_steps
            result["mean_gradient_norm"] = (
                float(np.mean(gradient_norms)) if gradient_norms else None
            )
        table = None
        if collect:
            ids, groups = identity['ids'], identity['groups']
            if len(predictions) != len(ids):
                raise ValueError(
                    "Evaluation sampler must visit every example exactly once"
                )
            table = PredictionTable(
                self.task,
                ids,
                np.concatenate(logits_list),
                references if references else None,
                groups,
                identity['content_keys'],
            )
            result['dataset_sha256'] = identity['sha256']
        return result, table

    def state_dict(self):
        return {
            "precision": self.precision.state_dict(),
            "schedule": self.schedule.state_dict()
            if self.schedule is not None
            else None,
        }

    def load_state_dict(self, state):
        self.precision.load_state_dict(state["precision"])
        if (state["schedule"] is None) != (self.schedule is None):
            raise ValueError("Resume scheduler differs from the training run")
        if self.schedule is not None:
            self.schedule.load_state_dict(state["schedule"])
