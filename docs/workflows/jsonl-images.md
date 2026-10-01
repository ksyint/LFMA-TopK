# Prepare local image benchmark bundles

Image JSONL rows contain `id`, `image`, and integer `label`. Relative image paths resolve from the source JSONL file and become absolute in the portable manifest. Use the selected benchmark's original class IDs.

```bash
python run.py manifest --task cifar10 --train data/cifar10/train.jsonl --validation data/cifar10/validation.jsonl --test data/cifar10/test.jsonl --output datasets/cifar10-manifest
python run.py train --config config.yaml --manifest-dir datasets/cifar10-manifest --device cuda
```

The loader verifies image paths, class ranges, stable split IDs, optional group separation, and the saved record fingerprint. Native ViT processing still runs in the collator. `manifest.json` records the task, split filenames, counts, label distribution, and digests.
