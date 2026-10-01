# cifar10 image benchmark

Ten image classes. Reserve a stratified 10% of the official training records for validation and keep the official test set separate.

```bash
python run.py prepare --task cifar10 --data-root datasets
python run.py train --config experiments/vit-base/cifar10/ratio_0p0003/alpha_12/ratio_0p0003-alpha_12-seed_42.yaml --data-root datasets --device cuda
```

The ViT processor converts images to RGB and applies the pretrained 224×224 resize, rescaling, and normalization. The classifier has 10 outputs. The adapter targets the query projection of every transformer block.

A local ImageFolder must provide `train`, `validation`, and `test` directories with identical class names. Select it with `--imagefolder`. Alternatively create a JSONL image bundle with `image`, integer `label`, and stable `id` fields and pass `--manifest-dir`. The manifest validator checks file existence, class-ID range, split identity, and optional group separation. Keep dataset partitions fixed across optimization seeds.
