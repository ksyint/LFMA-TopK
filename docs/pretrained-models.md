# Pretrained model files

Training resolves the selected model from Hugging Face and caches its weights, configuration, and preprocessing files. These four models are the encoders used by the experiment profiles.

| Setting | Model page | Portable directory |
| --- | --- | --- |
| `vit-base` | [ViT-B/16 ImageNet-21k](https://huggingface.co/google/vit-base-patch16-224-in21k) | `checkpoints/pretrained/vit-base/` |
| `vit-large` | [ViT-L/16 ImageNet-21k](https://huggingface.co/google/vit-large-patch16-224-in21k) | `checkpoints/pretrained/vit-large/` |
| `roberta-base` | [RoBERTa-Base](https://huggingface.co/FacebookAI/roberta-base) | `checkpoints/pretrained/roberta-base/` |
| `roberta-large` | [RoBERTa-Large](https://huggingface.co/FacebookAI/roberta-large) | `checkpoints/pretrained/roberta-large/` |

Run the matching preparation command once on a machine with network access. Each command downloads the pinned revision defined in `lfma/core.py`.

```bash
python run.py prepare --backbone vit-base --local-dir checkpoints/pretrained/vit-base
python run.py prepare --backbone vit-large --local-dir checkpoints/pretrained/vit-large
python run.py prepare --backbone roberta-base --local-dir checkpoints/pretrained/roberta-base
python run.py prepare --backbone roberta-large --local-dir checkpoints/pretrained/roberta-large
```

You can also download through the Hub CLI. This example fetches the complete ViT-B model files to the same directory.

```bash
hf download google/vit-base-patch16-224-in21k \
  --revision b4569560a39a0f1af58e3ddaf17facf20ab919b0 \
  --include model.safetensors config.json preprocessor_config.json \
  --local-dir checkpoints/pretrained/vit-base
```

For browser or direct-URL downloads, open the model's pinned file tree and place the listed files in its portable directory. Preserve file names.

| Backbone | Pinned file tree | Required files |
| --- | --- | --- |
| ViT-B/16 | [b4569560](https://huggingface.co/google/vit-base-patch16-224-in21k/tree/b4569560a39a0f1af58e3ddaf17facf20ab919b0) | `model.safetensors`, `config.json`, `preprocessor_config.json` |
| ViT-L/16 | [6074eaf2](https://huggingface.co/google/vit-large-patch16-224-in21k/tree/6074eaf2211423e928c93b93ef773d5da618aa7e) | `model.safetensors`, `config.json`, `preprocessor_config.json` |
| RoBERTa-Base | [e2da8e2f](https://huggingface.co/FacebookAI/roberta-base/tree/e2da8e2f811d1448a5b465c236feacd80ffbac7b) | `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` |
| RoBERTa-Large | [722cf37b](https://huggingface.co/FacebookAI/roberta-large/tree/722cf37b1afa9454edce342e7895e588b6ff1d59) | `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` |

The model loader checks encoder type, hidden width, and layer count against the selected backbone. The classification head is sized for the configured task, with one regression output for STS-B.

A prepared model works with automatic dataset downloads or fully local data:

```bash
python run.py train --config config.yaml --model-dir checkpoints/pretrained/vit-base
python run.py train --config experiments/roberta-large/mrpc/ratio_0p0005-alpha_150-seed_42.yaml \
  --model-dir checkpoints/pretrained/roberta-large --dataset-dir datasets/glue-mrpc --offline
```

`--offline` applies to both pretrained artifacts and datasets. CUDA remains the device for training, evaluation, and prediction. Evaluation and prediction accept the same `--model-dir` override when a checkpoint was moved to another machine.
