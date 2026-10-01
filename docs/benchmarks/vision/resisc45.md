# resisc45 image benchmark

Forty-five remote-sensing scene classes. Use the pinned parquet distribution with its separate train, validation, and test partitions.

```bash
python run.py prepare --task resisc45 --dataset-dir datasets/resisc45-arrow
python run.py train --config experiments/configs/catalog/vision/vit-base/resisc45/ratio_0p0003/alpha_12/seed_42.yaml --dataset-dir datasets/resisc45-arrow --device cuda
```

The ViT processor converts images to RGB and applies the pretrained 224×224 resize, rescaling, and normalization. The classifier has 45 outputs. The adapter targets the query projection of every transformer block.

A local ImageFolder must provide `train`, `validation`, and `test` directories with identical class names. Select it with `--imagefolder`. Alternatively create a JSONL image bundle with `image`, integer `label`, and stable `id` fields and pass `--manifest-dir`. The manifest validator checks file existence, class-ID range, split identity, and optional group separation. Keep dataset partitions fixed across optimization seeds.
