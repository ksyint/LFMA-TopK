# stanford_cars image benchmark

One hundred ninety-six vehicle categories. Download the pinned train/test parquet mirror and reserve validation examples from the original training set.

```bash
python run.py prepare --task stanford_cars --dataset-dir datasets/stanford_cars-arrow
python run.py train --config experiments/vit-base/stanford_cars/ratio_0p0003-alpha_12-seed_42.yaml --dataset-dir datasets/stanford_cars-arrow --device cuda
```

The ViT processor converts images to RGB and applies the pretrained 224×224 resize, rescaling, and normalization. The classifier has 196 outputs. The adapter targets the query projection of every transformer block.

A local ImageFolder must provide `train`, `validation`, and `test` directories with identical class names. Select it with `--imagefolder`. Alternatively create a JSONL image bundle with `image`, integer `label`, and stable `id` fields and pass `--manifest-dir`. The manifest validator checks file existence, class-ID range, split identity, and optional group separation. Keep dataset partitions fixed across optimization seeds.
