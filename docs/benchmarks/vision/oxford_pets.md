# oxford_pets image benchmark

Thirty-seven pet categories. Read category targets, retain the official test split, and reserve validation examples from trainval.

```bash
python run.py prepare --task oxford_pets --data-root datasets
python run.py train --config experiments/configs/catalog/vision/vit-base/oxford_pets/ratio_0p0003/alpha_12/seed_42.yaml --data-root datasets --device cuda
```

The ViT processor converts images to RGB and applies the pretrained 224×224 resize, rescaling, and normalization. The classifier has 37 outputs. The adapter targets the query projection of every transformer block.

A local ImageFolder must provide `train`, `validation`, and `test` directories with identical class names. Select it with `--imagefolder`. Alternatively create a JSONL image bundle with `image`, integer `label`, and stable `id` fields and pass `--manifest-dir`. The manifest validator checks file existence, class-ID range, split identity, and optional group separation. Keep dataset partitions fixed across optimization seeds.
