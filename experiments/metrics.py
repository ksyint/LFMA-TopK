import torch
from torch import nn


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss = correct = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        count += len(y)
    return {"loss": loss / count, "accuracy": correct / count, "samples": count}
