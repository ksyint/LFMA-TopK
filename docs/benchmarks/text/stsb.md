# STSB text benchmark

Use the official GLUE `stsb` records with fields `sentence1`, `sentence2`. Classification labels retain the dataset's integer IDs. STS-B uses floating similarity labels in [0,5].

```bash
python run.py prepare --task stsb --dataset-dir datasets/glue-stsb
python run.py train --config experiments/roberta-base/stsb/ratio_0p0005-alpha_150-seed_42.py --dataset-dir datasets/glue-stsb --device cuda
```

The selected RoBERTa profile tokenizes with its pinned pretrained tokenizer, truncates at the configured context budget, and dynamically pads each batch. The primary validation metric is `pearson`. The labeled validation split supplies local scores, while hidden-label test records produce predictions.

For prediction field examples, use [the stsb JSONL file](../../../examples/stsb.jsonl). A local text dataset can also be packaged with `python run.py manifest --task stsb --train train.jsonl --validation validation.jsonl --test test.jsonl --output datasets/stsb-manifest` and selected with `--manifest-dir`.
