"""CUDA autocast, gradient scaling, and accumulation step boundaries."""

from contextlib import nullcontext

import torch


class PrecisionPolicy:
    def __init__(self, name="fp32", max_grad_norm=1.0):
        if name not in ("fp32", "bf16", "fp16"):
            raise ValueError("Precision must be fp32, bf16, or fp16")
        if max_grad_norm <= 0:
            raise ValueError("Gradient clipping norm must be positive")
        self.name = name
        self.max_grad_norm = float(max_grad_norm)
        self.scaler = torch.amp.GradScaler("cuda", enabled=name == "fp16")
        self.updates = 0
        self.skipped = 0

    def context(self):
        if self.name == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.name == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)

    def backward(self, loss, divisor=1):
        if loss.ndim != 0 or not torch.isfinite(loss.detach()):
            raise RuntimeError("Training objective must be a finite scalar")
        if divisor < 1:
            raise ValueError("Accumulation divisor must be positive")
        self.scaler.scale(loss / divisor).backward()

    def step(self, optimizer, parameters, accumulated_examples=1):
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters were supplied")
        if accumulated_examples < 1:
            raise ValueError('An optimizer update requires accumulated examples')
        self.scaler.unscale_(optimizer)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(accumulated_examples)
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        if not self.scaler.is_enabled() and not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("Nonfinite unscaled gradient norm")
        old_scale = self.scaler.get_scale()
        self.scaler.step(optimizer)
        self.scaler.update()
        skipped = self.scaler.get_scale() < old_scale
        self.updates += int(not skipped)
        self.skipped += int(skipped)
        optimizer.zero_grad(set_to_none=True)
        return {
            "gradient_norm": float(norm),
            "skipped": skipped,
            "scale": self.scaler.get_scale(),
        }

    def state_dict(self):
        return {
            "name": self.name,
            "max_grad_norm": self.max_grad_norm,
            "scaler": self.scaler.state_dict(),
            "updates": self.updates,
            "skipped": self.skipped,
        }

    def load_state_dict(self, state):
        if state["name"] != self.name or state["max_grad_norm"] != self.max_grad_norm:
            raise ValueError("Precision settings changed during resume")
        self.scaler.load_state_dict(state["scaler"])
        self.updates = int(state["updates"])
        self.skipped = int(state["skipped"])


def accumulation_window(step, steps, accumulation):
    if accumulation < 1 or steps < 1 or not 0 <= step < steps:
        raise ValueError("Invalid accumulation coordinates")
    start = step // accumulation * accumulation
    count = min(accumulation, steps - start)
    return count, step + 1 == start + count


def move_batch(batch, device):
    if torch.device(device).type != "cuda":
        raise ValueError("Training batches require a CUDA device")
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch(value, device) for value in batch]
    return batch


def assert_cuda_model(model):
    devices = {parameter.device for parameter in model.parameters()}
    if not devices or any(device.type != "cuda" for device in devices):
        raise ValueError("Every model parameter must reside on CUDA")
    if len(devices) != 1:
        raise ValueError("This training loop expects one CUDA device per model")
    return next(iter(devices))


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0
