# COLA text benchmark

Use the official GLUE `cola` records with fields `sentence`. Classification labels retain the dataset's integer IDs. STS-B uses floating similarity labels in [0,5].

```bash
python run.py prepare --task cola --dataset-dir datasets/glue-cola
python run.py train --config experiments/roberta-base/cola/ratio_0p0005-alpha_150-seed_42.py --dataset-dir datasets/glue-cola --device cuda
```

The selected RoBERTa profile tokenizes with its pinned pretrained tokenizer, truncates at the configured context budget, and dynamically pads each batch. The primary validation metric is `matthews_correlation`. The labeled validation split supplies local scores, while hidden-label test records produce predictions.

For prediction field examples, use [the cola JSONL file](../../examples/prediction/cola.jsonl). A local text dataset can also be packaged with `python run.py manifest --task cola --train train.jsonl --validation validation.jsonl --test test.jsonl --output datasets/cola-manifest` and selected with `--manifest-dir`.
