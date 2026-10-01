# vision-ablation experiment selection

ViT models across all configured support ratios and scales.

```bash
python run.py grid --protocol vision-ablation --dry-run
python run.py plan --directory results/plans/vision-ablation --create --protocol vision-ablation --seeds 42 123 456
python run.py plan --directory results/plans/vision-ablation --execute --device cuda
```

Plan creation snapshots every resolved configuration. Execution records the state of each run, resumes its last complete epoch when present, selects the best validation adapter, and evaluates that adapter on the benchmark's scoring partition. A later execution continues unfinished runs.

Use `--tasks` and `--backbones` when creating the plan to narrow the requested benchmark. `status.json` and `events.jsonl` retain the per-run state and timestamps. `train.log` and `evaluate.log` live beside each run. The direct `python run.py protocol --protocol vision-ablation --device cuda` path remains available for a single sequential invocation.
