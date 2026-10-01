# SST2 text benchmark

Use the official GLUE `sst2` records with fields `sentence`. Classification labels retain the dataset's integer IDs. STS-B uses floating similarity labels in [0,5].

```bash
python run.py prepare --task sst2 --dataset-dir datasets/glue-sst2
python run.py train --config experiments/configs/catalog/glue/roberta-base/sst2/ratio_0p0005/alpha_150/seed_42.yaml --dataset-dir datasets/glue-sst2 --device cuda
```

The selected RoBERTa profile tokenizes with its pinned pretrained tokenizer, truncates at the configured context budget, and dynamically pads each batch. The primary validation metric is `accuracy`. The labeled validation split supplies local scores, while hidden-label test records produce predictions.

For prediction field examples, use [the sst2 JSONL file](../../../examples/prediction/sst2.jsonl). A local text dataset can also be packaged with `python run.py manifest --task sst2 --train train.jsonl --validation validation.jsonl --test test.jsonl --output datasets/sst2-manifest` and selected with `--manifest-dir`.
