"""The compact feature backbone used by train.py and eval.py."""
from torch import nn
import torch

from adapters import FourierLinear, inject_adapters, merge_adapters


class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def make_model(cfg):
    return FeatureMLP(**cfg["model"])
