"""Named optimizer groups with resumable schedule coordinates."""

import math

import torch


def parameter_groups(model, config):
    head_ids = {id(parameter) for parameter in model.classifier.parameters()}
    buckets = {}
    names = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = "head" if id(parameter) in head_ids else "adapter"
        decay = config["weight_decay"]
        if config.get("exclude_bias_decay", False) and (
            name.endswith(".bias") or parameter.ndim == 1
        ):
            decay = 0.0
        key = role, decay
        buckets.setdefault(key, []).append(parameter)
        names.setdefault(key, []).append(name)
    if not buckets:
        raise ValueError("Model exposes no trainable parameters")
    groups = []
    records = []
    for (role, decay), parameters in buckets.items():
        learning_rate = (
            config["head_learning_rate"] if role == "head" else config["learning_rate"]
        )
        if learning_rate <= 0 or decay < 0:
            raise ValueError(
                "Learning rates must be positive and weight decay nonnegative"
            )
        label = f"{role}-decay-{decay:g}"
        groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": decay,
                "name": label,
            }
        )
        records.append(
            {
                "name": label,
                "learning_rate": learning_rate,
                "weight_decay": decay,
                "parameters": sum(parameter.numel() for parameter in parameters),
                "tensors": names[(role, decay)],
            }
        )
    return groups, records


def build_optimizer(model, config):
    groups, records = parameter_groups(model, config)
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(config.get("betas", (0.9, 0.999))),
        eps=config.get("epsilon", 1e-8),
    )
    return optimizer, records


class UpdateSchedule:
    def __init__(
        self, optimizer, total_updates, kind="constant", warmup=0, minimum_ratio=0.0
    ):
        if total_updates < 1 or not 0 <= warmup < total_updates:
            raise ValueError("Warmup must be smaller than the positive update budget")
        if kind not in ("constant", "linear", "cosine") or not 0 <= minimum_ratio <= 1:
            raise ValueError("Unsupported learning-rate schedule")
        self.optimizer = optimizer
        self.total_updates = int(total_updates)
        self.kind = kind
        self.warmup = int(warmup)
        self.minimum_ratio = float(minimum_ratio)
        self.base_rates = [group["lr"] for group in optimizer.param_groups]
        self.updates = 0
        self.apply()

    def factor(self):
        if self.updates < self.warmup:
            return (self.updates + 1) / self.warmup
        progress = min(
            1.0, (self.updates - self.warmup) / max(1, self.total_updates - self.warmup)
        )
        if self.kind == "constant":
            return 1.0
        scale = (
            1 - progress
            if self.kind == "linear"
            else 0.5 * (1 + math.cos(math.pi * progress))
        )
        return self.minimum_ratio + (1 - self.minimum_ratio) * scale

    def apply(self):
        factor = self.factor()
        for base, group in zip(self.base_rates, self.optimizer.param_groups):
            group["lr"] = base * factor

    def step(self):
        self.updates += 1
        self.apply()

    def state_dict(self):
        return {
            "total_updates": self.total_updates,
            "kind": self.kind,
            "warmup": self.warmup,
            "minimum_ratio": self.minimum_ratio,
            "base_rates": self.base_rates,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in expected.keys() - {"updates"}:
            if state[key] != expected[key]:
                raise ValueError(f"Schedule resume changed {key}")
        self.updates = int(state["updates"])
        self.apply()


def update_budget(loader, config):
    if len(loader) == 0:
        raise ValueError("Training loader is empty")
    return math.ceil(len(loader) / config["gradient_accumulation"]) * config["epochs"]


def optimizer_summary(optimizer):
    return [
        {
            "name": group.get("name", str(index)),
            "learning_rate": group["lr"],
            "weight_decay": group["weight_decay"],
            "parameters": sum(parameter.numel() for parameter in group["params"]),
            "state_tensors": sum(
                len(optimizer.state.get(parameter, {})) for parameter in group["params"]
            ),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]
