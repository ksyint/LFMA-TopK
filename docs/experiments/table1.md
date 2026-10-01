# table1 experiment selection

ViT-B and ViT-L across seven image tasks with support ratio 0.05 and scale 12.

```bash
python run.py grid --protocol table1 --dry-run
python run.py plan --directory results/plans/table1 --create --protocol table1 --seeds 42 123 456
python run.py plan --directory results/plans/table1 --execute --device cuda
```

Plan creation snapshots every resolved configuration. Execution records the state of each run, resumes its last complete epoch when present, selects the best validation adapter, and evaluates that adapter on the benchmark's scoring partition. A later execution continues unfinished runs.

Use `--tasks` and `--backbones` when creating the plan to narrow the requested benchmark. `status.json` and `events.jsonl` retain the per-run state and timestamps. `train.log` and `evaluate.log` live beside each run. The direct `python run.py protocol --protocol table1 --device cuda` path remains available for a single sequential invocation.
