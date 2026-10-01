# Transfer an adapter archive

Create an archive containing the sparse adapter, classifier, metadata, and saved processor. Add optimizer state when the destination will continue training.

```bash
python run.py bundle --pack results/vit-base/cifar10/best --output exports/cifar10.tar.gz
python run.py bundle --inspect exports/cifar10.tar.gz --verify
python run.py bundle --unpack exports/cifar10.tar.gz --output checkpoints/imported/cifar10
```

Extraction accepts ordinary files and directories within the destination, checks expanded size, verifies every checksum, and validates the restored adapter. The matching pretrained backbone remains selected by its recorded model ID and revision. Use `--include-optimizer` during packing for a resumable archive.
