import torch
from torch import nn
import pytest
from adapters import FourierLinear, inject_adapters, merge_adapters


def test_full_spectrum_reconstructs_rectangular_update():
    torch.manual_seed(1)
    base = nn.Linear(7, 3)
    delta = torch.randn_like(base.weight)
    adapter = FourierLinear(base, delta, top_k_ratio=1.0, alpha=2.0)
    torch.testing.assert_close(adapter.delta_weight(), 2 * delta)
    x = torch.randn(2, 4, 7)
    torch.testing.assert_close(adapter(x), nn.functional.linear(x, base.weight + 2 * delta, base.bias))
    torch.testing.assert_close(adapter(x), adapter.merged()(x))


def test_sparse_reconstruction_and_stable_support():
    delta = torch.randn(4, 5)
    adapter = FourierLinear(nn.Linear(5, 4), delta, k=4, alpha=1)
    expected = torch.fft.fft2(delta)
    mask = torch.zeros(20, dtype=torch.bool)
    mask[adapter.indices] = True
    expected.flatten()[~mask] = 0
    torch.testing.assert_close(adapter.delta_weight(), torch.fft.ifft2(expected).real)
    before = adapter.indices.clone()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
    adapter(torch.randn(2, 5)).sum().backward()
    optimizer.step()
    assert torch.equal(before, adapter.indices)
