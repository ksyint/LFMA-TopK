# Validate portable adapter state

Adapter inspection reads the JSON configuration and safetensors header before any pretrained model is allocated. It checks task compatibility, exact projection names, support and classifier shapes, dtypes, and contiguous tensor byte ranges.

```bash
python run.py adapter --checkpoint results/vit-base/cifar10/best --hashes --write-integrity
python run.py adapter --checkpoint results/vit-base/cifar10/best --verify-integrity
```

`bundle_manifest.json` stores checksums and byte counts for the checkpoint artifacts. Verification compares both content hashes and the complete file inventory. Sparse tensor restoration also invokes the structural validator before loading the base model.
