# table2 experiment selection

RoBERTa-Base across six GLUE tasks and its three configured support ratios.

```bash
python run.py grid --protocol table2 --dry-run
python run.py plan --directory results/plans/table2 --create --protocol table2 --seeds 42 123 456
python run.py plan --directory results/plans/table2 --execute --device cuda
```

Plan creation snapshots every resolved configuration. Execution records the state of each run, resumes its last complete epoch when present, selects the best validation adapter, and evaluates that adapter on the benchmark's scoring partition. A later execution continues unfinished runs.

Use `--tasks` and `--backbones` when creating the plan to narrow the requested benchmark. `status.json` and `events.jsonl` retain the per-run state and timestamps. `train.log` and `evaluate.log` live beside each run. The direct `python run.py protocol --protocol table2 --device cuda` path remains available for a single sequential invocation.
