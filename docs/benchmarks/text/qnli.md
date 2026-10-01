# QNLI text benchmark

Use the official GLUE `qnli` records with fields `question`, `sentence`. Classification labels retain the dataset's integer IDs. STS-B uses floating similarity labels in [0,5].

```bash
python run.py prepare --task qnli --dataset-dir datasets/glue-qnli
python run.py train --config experiments/roberta-base/qnli/ratio_0p0005-alpha_150-seed_42.py --dataset-dir datasets/glue-qnli --device cuda
```

The selected RoBERTa profile tokenizes with its pinned pretrained tokenizer, truncates at the configured context budget, and dynamically pads each batch. The primary validation metric is `accuracy`. The labeled validation split supplies local scores, while hidden-label test records produce predictions.

For prediction field examples, use [the qnli JSONL file](../../../examples/prediction/qnli.jsonl). A local text dataset can also be packaged with `python run.py manifest --task qnli --train train.jsonl --validation validation.jsonl --test test.jsonl --output datasets/qnli-manifest` and selected with `--manifest-dir`.
