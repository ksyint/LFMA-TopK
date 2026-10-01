# vit-base pretrained artifacts

The `vit-base` setting uses `google/vit-base-patch16-224-in21k` at revision `b4569560a39a0f1af58e3ddaf17facf20ab919b0`. Its encoder width is 768 across 12 transformer blocks.

```bash
python run.py prepare --backbone vit-base --local-dir checkpoints/pretrained/vit-base
hf download google/vit-base-patch16-224-in21k --revision b4569560a39a0f1af58e3ddaf17facf20ab919b0 --local-dir checkpoints/pretrained/vit-base
python run.py train --config experiments/vit-base/cifar10/ratio_0p0003-alpha_12-seed_42.yaml --model-dir checkpoints/pretrained/vit-base --offline --device cuda
```

The first preparation command and the direct Hub command are alternatives. Keep `config.json`, every downloaded weight shard, the shard index when present, and the model's processor or tokenizer artifacts in that directory. Online training downloads the pinned source automatically when `--model-dir` is omitted.

Each adapter stores the resolved revision and task processor alongside its sparse coefficients and class head. Evaluation can restore the same base from the cache or a matching local directory. `python run.py bundle --deployment <adapter-directory>` prints the download and offline evaluation arguments for a saved adapter.
