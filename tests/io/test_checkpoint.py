import torch
from adapters import inject_adapters
from adapters.io import save_checkpoint, load_adapted_model
from model import make_model
from utils import load_config


def test_checkpoint_reconstructs_support_and_predictions(tmp_path):
    cfg = load_config('config.yaml')
    model = make_model(cfg).to("cuda")
    inject_adapters(model, seed=cfg['train']['seed'], **cfg['adapter'])
    x = torch.randn(3, cfg['model']['input_dim'], device='cuda')
    save_checkpoint({'config': cfg, 'model': model.state_dict()}, tmp_path, is_best=True)
    loaded, _ = load_adapted_model(tmp_path / 'checkpoint_best.pth.tar', device='cuda')
    torch.testing.assert_close(model(x), loaded(x))


def test_dotted_overrides_preserve_other_settings():
    cfg = load_config('config.yaml', ['train.epochs', '2', 'adapter.alpha', '5'])
    assert cfg['train']['epochs'] == 2
    assert cfg['adapter']['alpha'] == 5
    assert cfg['adapter']['top_k_ratio'] == 0.05
