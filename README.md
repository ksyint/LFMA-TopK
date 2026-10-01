# LFMA: Parameter-Efficient Fine-Tuning via Layerwise Fourier Masked Adapter with Top-k Frequency Selection

PyTorch implementation | [Paper](https://proceedings.mlr.press/v282/park26a.html)

## Summary

LFMA freezes the backbone and optimizes the largest-magnitude coefficients of an initialization update's Fourier transform. A real inverse FFT reconstructs the additive weight update. The support stays fixed and can be merged into the original layer for inference. Each complex slot stores two real trainable scalars.

The numerical adapter lives in `adapters/layers`, frequency selection in `adapters/spectral`, and checkpoint construction in `adapters/io`. Root `train.py` and `eval.py` run a compact feature-space experiment; `model.py` defines its frozen MLP backbone.

## 1. Installation

Use a CUDA-enabled PyTorch installation. Training, evaluation, and projection examples use CUDA. Select a device with `--device cuda` or `--device cuda:N`.

```bash
conda create -n lfma python=3.10
conda activate lfma
pip install -r requirements.txt
```

## 2. Training

All default settings are in [config.yaml](config.yaml). The feature NPZ contains `x_train`, `y_train`, `x_val`, `y_val`; inputs are finite floats `[N, input_dim]` and labels are integers `[N]`. Configure the dimensions and supply a pretrained `FeatureMLP.state_dict()`:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.yaml \
  --data features.npz --base-checkpoint pretrained_mlp.pth
```

Dotted overrides change individual settings, such as `--opts train.epochs 50 adapter.alpha 12`. Full-spectrum and sparse-support profiles are in `experiments/configs`. The optimizer updates only adapter coefficients. Checkpoints and measured metrics are saved under `train.save_dir`.

Continue an existing run with matching model/adapter settings and a larger total epoch count:

```bash
python train.py --data features.npz --resume results/lfma/checkpoint_last.pth.tar \
  --opts train.epochs 50
```

### Feature-space experiment grid

`experiments/configs/catalog` contains 216 complete configurations. They combine feature widths 384/512/768/1024, support ratios 0.0003/0.001/0.005/0.01/0.025/0.05, scales 12/60/120, and seeds 42/123/456. Every profile adapts both projections of a pretrained MLP with matching input/hidden width and 1,000 classes. Each configuration records its own output directory and works directly with `train.py --config`.

The grid runner filters the configurations and invokes the root training script. Match each width to its NPZ features and pretrained backbone using the path templates:

```bash
python -m experiments.run_grid --feature-dims 768 --ratios 0.001 --alphas 12 \
  --seeds 42 123 456 --data-template '/data/features_{input_dim}.npz' \
  --base-template '/weights/mlp_{input_dim}.pth' --device cuda
```

`--epochs` overrides the schedule, and `--output-root` changes the shared result directory. `python -m experiments.run_grid --dry-run` inspects all configuration schemas and reports the selected parameter budgets without running a model. The reproducible grid definition is in `experiments/grid.py`.

## 3. Evaluation

```bash
python eval.py --checkpoint results/lfma/checkpoint_best.pth.tar --data features.npz
```

Evaluation restores sparse support from the checkpoint and checks equality with a merged model. Tests cover rectangular transforms, frozen weights, support stability, checkpoint reconstruction, configuration overrides, and training updates.

## 4. Existing models

```python
import torch
from adapters import inject_adapters, merge_adapters

adapters = inject_adapters(
    model, target_names=["encoder.layer.0.attention.self.query"],
    top_k_ratio=0.05, alpha=12.0, init_std=1e-3, seed=42,
)
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad), lr=1e-4,
)
# Optimize your task loss, then fold the update into the backbone.
inference_model = merge_adapters(model)
```

`probes={module_name: delta_weight}` supplies custom spatial initialization. Otherwise seeded Gaussian updates are used. The API freezes all existing parameters; unfreeze a task head explicitly if required. Save adapter settings alongside the state dictionary and reinsert matching adapters before loading.

Weights follow PyTorch's `[out, in]` convention, with default backward-normalized FFTs and an unconstrained complex spectrum followed by real projection. `k = floor(out * in * ratio)`; an empty support is rejected. Use the adapter API with the pretrained model and task data appropriate to your experiment.
