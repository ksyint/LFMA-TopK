# LFMA: Parameter-Efficient Fine-Tuning via Layerwise Fourier Masked Adapter with Top-k Frequency Selection

Independent PyTorch implementation of LFMA | [Paper](https://proceedings.mlr.press/v282/park26a.html)

Junyoung Park, Soo Yong Kim, Sang Heon Lee, and Jeonghwan Lee. PMLR 282:415–427, 2026.

## Summary

LFMA freezes a linear layer and learns the largest-magnitude coefficients of an initialization update's 2D Fourier transform. The support stays fixed. A scaled real inverse FFT reconstructs the additive weight update, which can be merged for inference. Each retained complex coefficient is stored as two real trainable scalars.

`model.py` contains the adapter, exact-name module insertion, and merge utilities. They accept rectangular layers and inputs with arbitrary leading batch/sequence dimensions. The command-line example trains a frozen MLP on feature arrays and includes a deterministic synthetic task for checking optimization.

## Environmental Set-up

```bash
conda create -n lfma python=3.10
conda activate lfma
pip install -r requirements.txt
```

## Quick Start

```bash
python train.py --config configs/lfma.yaml --smoke
python eval.py --checkpoint results/lfma/best.pth
python -m pytest -q
```

For feature classification, provide an NPZ with `x_train`, `y_train`, `x_val`, `y_val`. Inputs are finite float arrays `[N, input_dim]`, labels are integers `[N]`. Set the dimensions in the YAML and supply a pretrained `FeatureMLP.state_dict()`:

```bash
python train.py --data features.npz --base-checkpoint pretrained_mlp.pth
python eval.py --checkpoint results/lfma/best.pth --data features.npz
```

Without a base checkpoint the example uses a frozen random feature backbone. `results/lfma/metrics.json` records the actual run; no paper benchmark numbers are claimed.

## Adapting an existing model

```python
from model import inject_adapters, merge_adapters

adapters = inject_adapters(
    model, target_names=["encoder.layer.0.attention.self.query"],
    top_k_ratio=0.05, alpha=12.0, init_std=1e-3, seed=42,
)
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad), lr=1e-4,
)
# Run your model's ordinary task loss and optimizer steps.
inference_model = merge_adapters(model)
```

Pass `probes={module_name: delta_weight}` to choose the spatial initialization explicitly. The API freezes every existing parameter, including output heads; unfreeze a task head explicitly if your protocol requires it. Save the model state and adapter settings together, and reinsert adapters with those settings before loading.

## Implementation notes

- Spatial initialization is configurable through `delta_W_init`; the example uses seeded Gaussian updates.
- `F.linear` uses PyTorch's `[out, in]` weight convention for both rectangular and square layers.
- The FFT uses PyTorch's default backward normalization, matching Algorithm 1. The real projection is applied without forcing conjugate-symmetric support.
- `k = floor(out * in * ratio)`; a zero-sized support raises an error. This avoids silently changing the requested budget.
- Transformer datasets, pretrained weights, and complete ViT/GLUE benchmark launchers are not bundled. This is a method implementation and feature-adaptation runner; published results have not been reproduced here.

## Citation

```bibtex
@inproceedings{park2026lfma,
  title={LFMA: Parameter-Efficient Fine-Tuning via Layerwise Fourier Masked Adapter with Top-k Frequency Selection},
  author={Park, Junyoung and Kim, Soo Yong and Lee, Sang Heon and Lee, Jeonghwan},
  booktitle={Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research},
  volume={282},
  pages={415--427},
  year={2026}
}
```
