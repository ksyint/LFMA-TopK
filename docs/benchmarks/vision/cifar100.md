# cifar100 image benchmark

One hundred fine-grained image classes. Use the fine labels and a stratified validation subset of the official training partition.

```bash
python run.py prepare --task cifar100 --data-root datasets
python run.py train --config experiments/configs/catalog/vision/vit-base/cifar100/ratio_0p0003/alpha_12/seed_42.yaml --data-root datasets --device cuda
```

The ViT processor converts images to RGB and applies the pretrained 224×224 resize, rescaling, and normalization. The classifier has 100 outputs. The adapter targets the query projection of every transformer block.

A local ImageFolder must provide `train`, `validation`, and `test` directories with identical class names. Select it with `--imagefolder`. Alternatively create a JSONL image bundle with `image`, integer `label`, and stable `id` fields and pass `--manifest-dir`. The manifest validator checks file existence, class-ID range, split identity, and optional group separation. Keep dataset partitions fixed across optimization seeds.
