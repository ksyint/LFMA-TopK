"""Run with python -m examples.frozen_projection from the repository root."""
import argparse
import torch
from torch import nn
from adapters import inject_adapters, merge_adapters
from utils import cuda_device


def main(args):
    args.device = cuda_device(args.device)
    torch.manual_seed(42)
    encoder = nn.Sequential(nn.Linear(12, 8), nn.Tanh(), nn.Linear(8, 4)).to(args.device)
    inject_adapters(encoder, ['0', '2'], top_k_ratio=0.25)
    optimizer = torch.optim.AdamW((p for p in encoder.parameters() if p.requires_grad), lr=0.05)
    x, target = torch.randn(32, 12, device=args.device), torch.randn(32, 4, device=args.device)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = (encoder(x) - target).square().mean()
        loss.backward()
        optimizer.step()
    torch.testing.assert_close(encoder(x), merge_adapters(encoder)(x))
    print(f'projection loss: {loss.item():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    main(parser.parse_args())
