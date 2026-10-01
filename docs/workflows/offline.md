# Offline execution

Prepare the selected pretrained model and dataset once, then keep their directories available on the CUDA host.

```bash
python run.py prepare --backbone vit-base --local-dir checkpoints/pretrained/vit-base
python run.py prepare --task cifar10 --data-root datasets
python run.py train --config config.yaml --model-dir checkpoints/pretrained/vit-base --data-root datasets --offline --device cuda
```

For GLUE, Cars, and RESISC45, a portable `DatasetDict.save_to_disk` directory is selected with `--dataset-dir`. JSONL bundles use `--manifest-dir`. Model loading, tokenization, and image processing retain the pinned pretrained identity and checkpoint processor.
