# Create and continue experiment plans

A plan stores immutable configuration snapshots and stable run IDs before training starts. A process lock prevents two executors from writing the same plan simultaneously.

```bash
python run.py plan --directory results/plans/cifar10 --create --protocol table1 --backbones vit-base --tasks cifar10 --seeds 42 123 456
python run.py plan --directory results/plans/cifar10 --execute --device cuda
python run.py plan --directory results/plans/cifar10
```

The last command inspects status. Execution continues the last completed checkpoint epoch for unfinished runs. Completed runs are skipped. `--keep-going` records an individual failure and continues the remaining plan. The per-run train/evaluation logs and event journal preserve the reason for each state transition.
