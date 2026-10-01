# roberta-large pretrained artifacts

The `roberta-large` setting uses `FacebookAI/roberta-large` at revision `722cf37b1afa9454edce342e7895e588b6ff1d59`. Its encoder width is 1024 across 24 transformer blocks.

```bash
python run.py prepare --backbone roberta-large --local-dir checkpoints/pretrained/roberta-large
hf download FacebookAI/roberta-large --revision 722cf37b1afa9454edce342e7895e588b6ff1d59 --local-dir checkpoints/pretrained/roberta-large
python run.py train --config experiments/configs/catalog/glue/roberta-large/mrpc/ratio_0p0001/alpha_150/seed_42.yaml --model-dir checkpoints/pretrained/roberta-large --offline --device cuda
```

The first preparation command and the direct Hub command are alternatives. Keep `config.json`, every downloaded weight shard, the shard index when present, and the model's processor or tokenizer artifacts in that directory. Online training downloads the pinned source automatically when `--model-dir` is omitted.

Each adapter stores the resolved revision and task processor alongside its sparse coefficients and class head. Evaluation can restore the same base from the cache or a matching local directory. `python run.py bundle --deployment <adapter-directory>` prints the download and offline evaluation arguments for a saved adapter.
