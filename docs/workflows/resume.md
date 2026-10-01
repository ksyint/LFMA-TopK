# Continue trained adapters

`last/` stores the completed epoch, sparse adapter, task head, optimizer, and Torch/CUDA random state. Resume selects a larger total epoch count while retaining the same model, task, and support configuration.

```bash
python run.py train --resume results/vit-base/cifar10/last --opts train.epochs 100 --device cuda
python run.py adapter --checkpoint results/vit-base/cifar10/last --hashes
```

Existing experiment plans derive their resume checkpoint from the saved run directory. Reuse the original dataset partition and preprocessing. The task head and exact support indices are restored before the optimizer state is loaded.
