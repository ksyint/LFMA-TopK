import torch
from torch.utils.data import DataLoader
from adapters import inject_adapters
from experiments.data import dataset
from model import make_model
from train import configure_optimizer, run_epoch
from utils import load_config


def test_training_epoch_only_updates_adapter():
    torch.manual_seed(42)
    cfg = load_config('config.yaml')
    model = make_model(cfg)
    adapters = inject_adapters(model, **cfg['adapter'])
    before = {name: adapter.base_layer.weight.clone() for name, adapter in adapters.items()}
    loader = DataLoader(dataset(cfg), batch_size=64)
    metrics = run_epoch(model, loader, 'cpu', configure_optimizer(model, cfg['train']))
    assert metrics['samples'] == 256 and metrics['loss'] > 0
    for name, adapter in adapters.items():
        torch.testing.assert_close(adapter.base_layer.weight, before[name])
