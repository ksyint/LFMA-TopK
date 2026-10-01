# vit-large pretrained artifacts

The `vit-large` setting uses `google/vit-large-patch16-224-in21k` at revision `6074eaf2211423e928c93b93ef773d5da618aa7e`. Its encoder width is 1024 across 24 transformer blocks.

```bash
python run.py prepare --backbone vit-large --local-dir checkpoints/pretrained/vit-large
hf download google/vit-large-patch16-224-in21k --revision 6074eaf2211423e928c93b93ef773d5da618aa7e --local-dir checkpoints/pretrained/vit-large
python run.py train --config experiments/configs/catalog/vision/vit-large/cifar10/ratio_0p0003/alpha_12/seed_42.yaml --model-dir checkpoints/pretrained/vit-large --offline --device cuda
```

The first preparation command and the direct Hub command are alternatives. Keep `config.json`, every downloaded weight shard, the shard index when present, and the model's processor or tokenizer artifacts in that directory. Online training downloads the pinned source automatically when `--model-dir` is omitted.

Each adapter stores the resolved revision and task processor alongside its sparse coefficients and class head. Evaluation can restore the same base from the cache or a matching local directory. `python run.py bundle --deployment <adapter-directory>` prints the download and offline evaluation arguments for a saved adapter.
