"""Train LFMA on frozen MLP features; the adapter API supports arbitrary models."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import inject_adapters
from utils import dataset, evaluate, load_config, make_model, set_seed, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lfma.yaml")
    parser.add_argument("--data", help="NPZ feature dataset; omitted = deterministic synthetic task")
    parser.add_argument("--base-checkpoint", help="FeatureMLP state_dict before adapter insertion")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg["train"].update(epochs=3, learning_rate=0.05, batch_size=64)
        cfg["adapter"].update(top_k_ratio=0.5, alpha=12.0)
    set_seed(cfg["train"]["seed"])
    torch.set_num_threads(cfg["train"].get("num_threads", 1))
    model = make_model(cfg).to(args.device)
    if args.base_checkpoint:
        model.load_state_dict(torch.load(args.base_checkpoint, map_location=args.device, weights_only=True))
    adapters = inject_adapters(model, seed=cfg["train"]["seed"], **cfg["adapter"])
    train_loader = DataLoader(dataset(cfg, args.data, "train"), batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset(cfg, args.data, "val"), batch_size=128)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    initial = evaluate(model, train_loader, args.device)
    history = []
    best = float("inf")
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x.to(args.device)), y.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
        scores = evaluate(model, val_loader, args.device)
        history.append({"epoch": epoch + 1, **scores})
        checkpoint = {"config": cfg, "model": model.state_dict(), "epoch": epoch + 1,
                      "optimizer": optimizer.state_dict(), "metrics": scores}
        torch.save(checkpoint, save_dir / "last.pth")
        if scores["loss"] < best:
            best = scores["loss"]
            torch.save(checkpoint, save_dir / "best.pth")
        print(f"epoch {epoch + 1}: {scores}")
    final = evaluate(model, train_loader, args.device)
    report = {"task": "npz_features" if args.data else "synthetic_sanity", "initial_train": initial,
              "final_train": final, "history": history,
              "trainable_real_scalars": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "support_sizes": {name: adapter.k for name, adapter in adapters.items()}}
    write_json(save_dir / "metrics.json", report)
    if args.smoke and final["loss"] >= initial["loss"]:
        raise RuntimeError("smoke optimization did not reduce training loss")


if __name__ == "__main__":
    main()
