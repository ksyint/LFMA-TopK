# Export a standard Transformers model

A merged export folds each Fourier update into the pretrained projection and retains the learned classifier. The output includes the task processor and a merge manifest.

```bash
python run.py evaluate --checkpoint results/vit-base/cifar10/best --split test --merge results/merged-vit --device cuda
```

Image exports load with `AutoModelForImageClassification.from_pretrained`. Text exports load with `AutoModelForSequenceClassification.from_pretrained`. Supply CUDA placement in the consuming application. `lfma_merge.json` records the original base identity, selected epoch, targets, and exported file sizes.
