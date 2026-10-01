# RTE text benchmark

Use the official GLUE `rte` records with fields `sentence1`, `sentence2`. Classification labels retain the dataset's integer IDs. STS-B uses floating similarity labels in [0,5].

```bash
python run.py prepare --task rte --dataset-dir datasets/glue-rte
python run.py train --config experiments/configs/catalog/glue/roberta-base/rte/ratio_0p0005/alpha_150/seed_42.yaml --dataset-dir datasets/glue-rte --device cuda
```

The selected RoBERTa profile tokenizes with its pinned pretrained tokenizer, truncates at the configured context budget, and dynamically pads each batch. The primary validation metric is `accuracy`. The labeled validation split supplies local scores, while hidden-label test records produce predictions.

For prediction field examples, use [the rte JSONL file](../../../examples/prediction/rte.jsonl). A local text dataset can also be packaged with `python run.py manifest --task rte --train train.jsonl --validation validation.jsonl --test test.jsonl --output datasets/rte-manifest` and selected with `--manifest-dir`.
