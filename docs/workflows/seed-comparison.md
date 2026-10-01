# Compare repeated-seed benchmarks

Comparison discovers completed best-adapter evaluations and groups them by backbone, task, support ratio, and scale. A setting records its seeds, missing expected seeds, mean, median, standard deviation, and parameter count.

```bash
python run.py compare --root results/catalog --output reports/catalog --expected-seeds 42 123 456 789 1024
```

The Pareto table identifies settings for which no measured alternative has both fewer adapter scalars and a higher mean task score. For paired comparisons, use `--baseline-root`, choose one ratio/scale on each side, and retain identical backbone/task/seed coordinates and pretrained revisions. Paired differences can be bootstrapped over those seeds.
