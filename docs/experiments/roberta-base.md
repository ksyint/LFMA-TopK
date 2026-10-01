# roberta-base pretrained artifacts

The `roberta-base` setting uses `FacebookAI/roberta-base` at revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. Its encoder width is 768 across 12 transformer blocks.

```bash
python run.py prepare --backbone roberta-base --local-dir checkpoints/pretrained/roberta-base
hf download FacebookAI/roberta-base --revision e2da8e2f811d1448a5b465c236feacd80ffbac7b --local-dir checkpoints/pretrained/roberta-base
python run.py train --config experiments/configs/catalog/glue/roberta-base/mrpc/ratio_0p0005/alpha_150/seed_42.yaml --model-dir checkpoints/pretrained/roberta-base --offline --device cuda
```

The first preparation command and the direct Hub command are alternatives. Keep `config.json`, every downloaded weight shard, the shard index when present, and the model's processor or tokenizer artifacts in that directory. Online training downloads the pinned source automatically when `--model-dir` is omitted.

Each adapter stores the resolved revision and task processor alongside its sparse coefficients and class head. Evaluation can restore the same base from the cache or a matching local directory. `python run.py bundle --deployment <adapter-directory>` prints the download and offline evaluation arguments for a saved adapter.
