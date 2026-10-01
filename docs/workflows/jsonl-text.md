# Prepare local text benchmark bundles

Text partitions use the selected GLUE field names and original labels. Every row has a stable unique ID. Optional group IDs must remain within a single partition.

```bash
python run.py manifest --task sst2 --train data/sst2/train.jsonl --validation data/sst2/validation.jsonl --test data/sst2/test.jsonl --output datasets/sst2-manifest
python run.py manifest --inspect datasets/sst2-manifest
```

Pass `--manifest-dir datasets/sst2-manifest` to the matching RoBERTa training profile. Native tokenization and dynamic padding are unchanged. Hidden-label text test records may omit labels or use -1. The files under `examples/manifests` illustrate the schema with names such as `sst2-train.jsonl` and `mrpc-validation.jsonl`. The preparation command accepts each source path separately and writes the bundle's standard `train.jsonl`, `validation.jsonl`, and `test.jsonl` names. Prepare the actual benchmark partitions for training and scoring.
