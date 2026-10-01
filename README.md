# LFMA: Parameter-Efficient Fine-Tuning via Layerwise Fourier Masked Adapter with Top-k Frequency Selection

LFMA freezes a pretrained transformer and optimizes a fixed set of large-magnitude Fourier coefficients. Each complex coefficient uses two real scalars. A real inverse FFT reconstructs the layer update. The task classifier is learned alongside the adapter and is included in every checkpoint.

## 1. Installation

Use Python 3.10+ and a CUDA-enabled PyTorch/torchvision installation matched to your GPU, then install the experiment dependencies:

```bash
conda create -n lfma python=3.10
conda activate lfma
pip install -r requirements.txt
```

Model training, evaluation, and prediction require CUDA. Choose a card with `CUDA_VISIBLE_DEVICES` or `--device cuda:N`. Fourier coefficients stay FP32. `--opts train.precision bf16` enables autocasting around transformer execution.

## 2. Pretrained models and benchmark data

The first training command downloads the requested pretrained weights and image processor/tokenizer automatically. Model revisions are pinned in `experiments/protocols/models.py` and recorded with each adapter.

| Backbone setting | Pretrained weights | Adapted projections |
| --- | --- | --- |
| `vit-base` | [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) | Query in all 12 blocks |
| `vit-large` | [google/vit-large-patch16-224-in21k](https://huggingface.co/google/vit-large-patch16-224-in21k) | Query in all 24 blocks |
| `roberta-base` | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) | Query and value in all 12 blocks |
| `roberta-large` | [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) | Query and value in all 24 blocks |

ViT uses the ImageNet-21k pretrained B/16 or L/16 encoder and its saved 224×224 preprocessing. RoBERTa uses the pretrained tokenizer with dynamic padding and truncation at 512 tokens. A new task-sized classifier is initialized and trained. STS-B uses a one-output regression head.

| Task names | Runtime dataset acquisition |
| --- | --- |
| `sst2`, `mrpc`, `qnli`, `rte`, `cola`, `stsb` | [GLUE](https://huggingface.co/datasets/nyu-mll/glue), through Hugging Face Datasets |
| `cifar10`, `cifar100`, `oxford_pets`, `fgvc_aircraft`, `eurosat` | torchvision downloads and verifies the dataset archives |
| `resisc45` | [PyTorch Image Models parquet distribution](https://huggingface.co/datasets/timm/resisc45), including its train/validation/test splits |
| `stanford_cars` | [Stanford Cars parquet mirror](https://huggingface.co/datasets/tanganke/stanford_cars), original train/test image shards |

CIFAR, Pets, and Cars reserve a stratified 10% of the official training partition for validation. Their official test partitions stay separate. Aircraft and RESISC45 use their provided validation/test partitions. EuroSAT uses a seeded stratified 70/10/20 split. `data.split_seed` fixes image partitions across optimization seeds.

Models are cached in `.cache/huggingface`, Hub datasets in `.cache/datasets`, and torchvision archives in `datasets`. To prepare a portable local model and dataset before training:

```bash
python prepare.py --backbone roberta-base --local-dir checkpoints/pretrained/roberta-base
python prepare.py --task mrpc --dataset-dir datasets/glue-mrpc
python prepare.py --task cifar10 --data-root datasets
```

`--model-dir` loads a pretrained Transformers directory, `--dataset-dir` loads a Hugging Face `DatasetDict.save_to_disk` directory, and `--imagefolder` loads `train/`, `validation/`, and `test/` class folders with matching class names. `--offline` uses only existing files and caches. Model weights can also be fetched directly from the linked model pages into the corresponding `checkpoints/pretrained/<backbone>` directory with `prepare.py --backbone <backbone> --local-dir <directory>`.

### Local data layout and preprocessing

`prepare.py` keeps raw examples and labels. Pixel normalization and text tokenization run in the collator using the pretrained processor saved with the checkpoint. Image inputs convert to RGB, resize to 224×224, rescale to [0,1], and normalize with the ViT processor's mean/std of 0.5 per channel.

Torchvision prepares these paths under `--data-root`:

| Dataset | Prepared directory |
| --- | --- |
| CIFAR-10 | `cifar-10-batches-py/` |
| CIFAR-100 | `cifar-100-python/` |
| Oxford Pets | `oxford-iiit-pet/images/` and `annotations/` |
| FGVC Aircraft | `fgvc-aircraft-2013b/data/` |
| EuroSAT | `eurosat/2750/` |

For portable Hub image datasets, save the raw image/label records first and point training at that directory:

```bash
python prepare.py --task resisc45 --dataset-dir datasets/resisc45-arrow
python train.py --config experiments/configs/catalog/vision/vit-base/resisc45/ratio_0p0500/alpha_12/seed_42.yaml \
  --dataset-dir datasets/resisc45-arrow
```

A local ImageFolder directory has `train/<class>/*`, `validation/<class>/*`, and `test/<class>/*`. Every split uses the same class names and alphabetical class-index mapping. Choose a benchmark configuration with the matching class count, then pass `--imagefolder /data/prepared-images`.

GLUE JSONL prediction records use these fields. Supervised local DatasetDict records additionally include `label`:

| Task | Text columns | Label |
| --- | --- | --- |
| SST-2, CoLA | `sentence` | Integer class ID |
| MRPC, RTE | `sentence1`, `sentence2` | Integer class ID |
| QNLI | `question`, `sentence` | Integer class ID |
| STS-B | `sentence1`, `sentence2` | Floating similarity score |

The tokenizer adds model special tokens, truncates to `data.max_length`, and pads to the longest record in each batch. `prepare.py --task <task> --dataset-dir <path>` produces an Arrow DatasetDict containing `train`, `validation`, and `test`. Evaluation writes `metrics.json` and `predictions.tsv` with columns `index` and `prediction`.

For model-specific local directories, pinned download commands, and the files needed for manual staging, see [pretrained artifacts](docs/pretrained-models.md). [Dataset preparation](docs/datasets.md) gives split sizes, cached paths, and portable-record examples.

## 3. Training

The default configuration trains ImageNet-21k ViT-B/16 on CIFAR-10 with query adapters, ratio 0.05, scale 12, AdamW at 1e-4, weight decay 1e-4, batch size 32, and a trainable class head:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.yaml
```

Train an actual RoBERTa GLUE experiment from a complete profile:

```bash
python train.py --config experiments/configs/catalog/glue/roberta-base/mrpc/ratio_0p0016/alpha_150/seed_42.yaml
```

For an offline run with prepared artifacts:

```bash
python train.py --config experiments/configs/catalog/glue/roberta-base/mrpc/ratio_0p0016/alpha_150/seed_42.yaml \
  --model-dir checkpoints/pretrained/roberta-base --dataset-dir datasets/glue-mrpc --offline
```

Dotted overrides change individual settings. For example, `--opts train.batch_size 8 train.gradient_accumulation 4` keeps an effective batch of 32. `train.learning_rate` controls Fourier coefficients. `train.head_learning_rate` controls the task head. Learning rates remain constant over the configured epoch schedule.

Each run writes `history.jsonl`, measured metrics, and `last/` and `best/` adapter directories. Each directory contains sparse coefficients and support indices, the complete task head, the pinned base-model identity, saved preprocessing, and optimizer/random state. Resume with a larger total epoch count:

```bash
python train.py --resume results/vit-base/cifar10/last --opts train.epochs 100
```

## 4. Experiment grid

The catalog contains 600 complete configurations. The 420 vision runs cover two ViTs, seven datasets, three support ratios, two scales, and five seeds. The 180 GLUE runs cover two RoBERTas, six tasks, three backbone-specific support ratios, and five seeds. Vision scales are 12 and 120. Ratios and seeds live in `experiments/grid.py`, and every profile constructs the selected pretrained transformer.

```bash
python -m experiments.run_grid --backbones vit-base --tasks cifar10 oxford_pets \
  --ratios 0.05 --alphas 12 --seeds 42 123 456 --device cuda
python -m experiments.run_grid --backbones roberta-large --tasks cola stsb --ratios 0.0005 --seeds 42
```

`--model-template '/weights/{backbone}'` selects local pretrained directories. `--dataset-template '/datasets/{task}'` selects saved Hub datasets, and `--imagefolder-template '/images/{task}'` selects image class folders. `--offline` reuses local artifacts. `--output-root` and `--epochs` override the result location and duration.

Named protocols select the settings used in the paper's tables. `run_protocol.py` trains each selected profile, restores its best validation checkpoint, and scores the image test set or labeled GLUE validation set:

```bash
python run_protocol.py --protocol table1 --backbones vit-base --tasks cifar10
python run_protocol.py --protocol table2 --tasks mrpc
python run_protocol.py --protocol table3 --tasks cola stsb
python summarize.py --root results/catalog --output results/summary
```

The summary records each seed's result and exports the median, mean, sample standard deviation, and missing seeds as JSON and CSV. `vision-sparse` selects ratio 0.0003 with scale 120. `vision-ablation` selects all vision support/scale combinations. See [experiment settings](docs/experiments.md) for the full selection rules.

Configuration inspection does not load model tensors or fetch artifacts:

```bash
python -m experiments.run_grid --protocol table1 --dry-run
python inspect_adapter.py --config config.yaml
python inspect_adapter.py --checkpoint results/vit-base/cifar10/best
```

The inspector computes encoder, task-head, and sparse-adapter budgets from the pinned architecture. For a checkpoint it also reads tensor shapes directly from the safetensors header and checks the saved trainable counts.

## 5. Evaluation, prediction, and merged export

```bash
python eval.py --checkpoint results/vit-base/cifar10/best --split test
python eval.py --checkpoint results/vit-base/cifar10/best --split test --merge results/merged-vit
python predict.py --checkpoint results/vit-base/cifar10/best --images image1.png image2.png
```

Evaluation records accuracy for classification, Matthews correlation for CoLA, and Pearson/Spearman correlation for STS-B. MRPC additionally records F1. GLUE's labeled validation split supports local scoring. `--split test` writes test predictions to `predictions.tsv` for tasks with hidden labels. Text prediction accepts `--text records.jsonl` containing the same sentence fields as the selected GLUE task.

A merged export is a standard Transformers model directory containing the adapted encoder, learned head, and preprocessing. It loads with `AutoModelForImageClassification.from_pretrained` or `AutoModelForSequenceClassification.from_pretrained` without LFMA wrappers. `lfma_merge.json` records the base revision, task, adapted projections, selected epoch, and exported file sizes.

## 6. Adapter modules

`adapters/layers` implements the sparse update, `adapters/spectral` selects fixed support, and `adapters/io` stores portable pretrained-model adapters. `experiments/tasks` owns backbone/task construction. `experiments/data/benchmarks` loads raw examples. `experiments/optimization` owns AdamW and supervised epoch execution. `experiments/evaluation/reports` collects completed seed runs. Root scripts keep training, evaluation, preparation, and prediction explicit. See [module and checkpoint layout](docs/implementation.md) for the data flow.

The adapter API accepts exact `nn.Linear` names and optional spatial update probes. Weights use PyTorch's `[out, in]` layout and default backward-normalized FFTs. `k = floor(out * in * ratio)` and each coefficient costs two real trainable scalars. Merging folds the learned update into each frozen projection.
