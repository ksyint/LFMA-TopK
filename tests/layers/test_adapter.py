import torch
from torch import nn
import pytest
from adapters import FourierLinear, inject_adapters, merge_adapters


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
