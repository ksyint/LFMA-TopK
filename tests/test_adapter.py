"""Sparse reconstruction, frozen bases, and merged adapter invariants."""

import torch
import pytest
from torch import nn
from lfma.adaptation.layers import FourierLinear
from lfma.adaptation.injection import inject_adapters, merge_adapters


def test_only_selected_coefficients_train_and_base_frozen():
    base = nn.Linear(8, 6)
    adapter = FourierLinear(base, torch.randn_like(base.weight), k=5)
    adapter(torch.randn(10, 8)).square().sum().backward()
    assert adapter.c.shape == (5, 2)
    assert adapter.c.grad.abs().sum() > 0
    assert base.weight.grad is None and not base.weight.requires_grad
    spectrum = torch.fft.fft2(adapter.delta_weight() / adapter.alpha)
    # Real reconstruction creates conjugate partners. the stored sparse spectrum
    # still has exactly k entries. Do not incorrectly assert spectral sparsity of Re.
    assert torch.isfinite(spectrum).all()


def test_nested_injection_and_merge():
    model = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 2))
    inject_adapters(model, ["0", "2"], top_k_ratio=0.5)
    x = torch.randn(3, 5)
    torch.testing.assert_close(model(x), merge_adapters(model)(x))
    assert not any(p.requires_grad for p in model[0].base_layer.parameters())


def test_invalid_empty_support_rejected():
    with pytest.raises(ValueError):
        FourierLinear(nn.Linear(2, 2), torch.zeros(2, 2), top_k_ratio=0.01)


def test_full_spectrum_reconstructs_rectangular_update():
    torch.manual_seed(1)
    base = nn.Linear(7, 3)
    delta = torch.randn_like(base.weight)
    adapter = FourierLinear(base, delta, top_k_ratio=1.0, alpha=2.0)
    torch.testing.assert_close(adapter.delta_weight(), 2 * delta)
    x = torch.randn(2, 4, 7)
    torch.testing.assert_close(
        adapter(x), nn.functional.linear(x, base.weight + 2 * delta, base.bias)
    )
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
