"""Evaluate a saved adapter and verify merged-inference equivalence."""
import argparse

import torch
from torch.utils.data import DataLoader

from model import inject_adapters, merge_adapters
from utils import dataset, evaluate, make_model, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/eval.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    cfg = checkpoint["config"]
    model = make_model(cfg).to(args.device)
    inject_adapters(model, seed=cfg["train"]["seed"], **cfg["adapter"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(dataset(cfg, args.data, "val"), batch_size=128)
    metrics = evaluate(model, loader, args.device)
    merged = merge_adapters(model).eval()
    with torch.no_grad():
        x = next(iter(loader))[0].to(args.device)
        metrics["merge_max_abs_error"] = (model(x) - merged(x)).abs().max().item()
        torch.testing.assert_close(model(x), merged(x), atol=1e-5, rtol=1e-5)
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
