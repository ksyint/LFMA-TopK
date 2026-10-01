# Measure learned Fourier support

Spectral inspection loads adapter coefficients onto CUDA and reconstructs one layer update at a time. It reports coefficient energy, support density, effective frequency count, conjugate coverage, radial energy, and the real spatial update norm.

```bash
python run.py spectrum --checkpoint results/vit-base/cifar10/best --output reports/cifar10-spectrum --bins 8 --top 20 --device cuda
```

CSV outputs contain per-layer summaries, radial bins, and the largest coefficients. Add `--compare <other-adapter>` to measure support overlap and Jaccard similarity for matching pretrained targets. The separately reported task-head statistics keep classifier parameters distinct from spectral coefficients.
