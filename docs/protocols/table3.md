# table3 experiment selection

RoBERTa-Large across six GLUE tasks and its three configured support ratios.

```bash
python run.py grid --protocol table3 --dry-run
python run.py plan --directory results/plans/table3 --create --protocol table3 --seeds 42 123 456
python run.py plan --directory results/plans/table3 --execute --device cuda
```

Plan creation snapshots every resolved configuration. Execution records the state of each run, resumes its last complete epoch when present, selects the best validation adapter, and evaluates that adapter on the benchmark's scoring partition. A later execution continues unfinished runs.

Use `--tasks` and `--backbones` when creating the plan to narrow the requested benchmark. `status.json` and `events.jsonl` retain the per-run state and timestamps. `train.log` and `evaluate.log` live beside each run. The direct `python run.py protocol --protocol table3 --device cuda` path remains available for a single sequential invocation.
