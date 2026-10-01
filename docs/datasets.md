# Dataset preparation

The experiment loaders consume raw images or text records. Preparation downloads and caches those examples. Model-specific transforms run at batch collation so that the same saved processor can be used for training, evaluation, and prediction.

## Image benchmarks

```bash
python run.py prepare --task cifar10 --data-root datasets
python run.py prepare --task cifar100 --data-root datasets
python run.py prepare --task oxford_pets --data-root datasets
python run.py prepare --task fgvc_aircraft --data-root datasets
python run.py prepare --task eurosat --data-root datasets
python run.py prepare --task resisc45 --dataset-dir datasets/resisc45-arrow
python run.py prepare --task stanford_cars --dataset-dir datasets/stanford-cars-arrow
```

| Task | Acquisition | Split handling |
| --- | --- | --- |
| CIFAR-10 | [torchvision CIFAR10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html) | Stratified 45,000 train, 5,000 validation, official 10,000 test |
| CIFAR-100 | [torchvision CIFAR100](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html) | Stratified 45,000 train, 5,000 validation, official 10,000 test |
| Oxford Pets | [torchvision OxfordIIITPet](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html) | Stratified 90/10 split of trainval, official test |
| FGVC Aircraft | [torchvision FGVCAircraft](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FGVCAircraft.html) | Variant labels and provided train/val/test splits |
| EuroSAT | [torchvision EuroSAT](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html) | Stratified 70/10/20 split using `data.split_seed` |
| RESISC45 | [timm/resisc45](https://huggingface.co/datasets/timm/resisc45) | Provided 18,900 train, 6,300 validation, 6,300 test |
| Stanford Cars | [tanganke/stanford_cars](https://huggingface.co/datasets/tanganke/stanford_cars) | Original train/test shards, stratified training holdout |

The torchvision loaders download and verify their archives under `--data-root`. The Hub loaders pin dataset revisions in `lfma/models/fourier/core.py`. Cars loads only `data/train-*.parquet` and `data/test-*.parquet`. Portable Hub datasets store `image` and integer `label` columns in a DatasetDict directory.

Images convert to RGB, resize to the saved ViT processor's 224×224 input, rescale pixels to [0,1], and normalize each channel with the processor's mean and standard deviation. Labels use the dataset's integer mapping. The two pinned ViTs use mean/std 0.5. The default pipeline keeps preprocessing identical across all image partitions.

To use prepared class folders, arrange the files as follows and supply `--imagefolder /data/cifar10` with a CIFAR-10 configuration:

```text
/data/cifar10/
  train/airplane/*.png
  train/automobile/*.png
  validation/airplane/*.png
  validation/automobile/*.png
  test/airplane/*.png
  test/automobile/*.png
```

Include all ten CIFAR-10 classes in each split. Other benchmarks use their own configured class count. ImageFolder assigns indices in alphabetical class-name order, and the loader verifies that every split has the same mapping.

## GLUE text data

```bash
python run.py prepare --task mrpc --dataset-dir datasets/glue-mrpc
python run.py prepare --task sst2 --dataset-dir datasets/glue-sst2
```

The same command accepts `qnli`, `rte`, `cola`, and `stsb`. It fetches the task subset from [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) at the pinned dataset revision and saves train, validation, and test partitions. Pass the directory to `--dataset-dir` when training or evaluating offline.

| Task | Required text fields | Supervised `label` |
| --- | --- | --- |
| SST-2, CoLA | `sentence` | 0 or 1 |
| MRPC, RTE | `sentence1`, `sentence2` | 0 or 1 |
| QNLI | `question`, `sentence` | 0 or 1 |
| STS-B | `sentence1`, `sentence2` | Floating score from 0 to 5 |

A DatasetDict made from local JSONL files can use the same schema:

```python
from datasets import load_dataset

records = load_dataset('json', data_files={
    'train': '/data/mrpc/train.jsonl',
    'validation': '/data/mrpc/validation.jsonl',
    'test': '/data/mrpc/test.jsonl',
})
records.save_to_disk('datasets/glue-mrpc')
```

An MRPC row is `{"sentence1": "First sentence.", "sentence2": "Second sentence.", "label": 1}`. Preserve the task's label mapping when preparing local data. The tokenizer inserts RoBERTa special tokens, handles sentence pairs, truncates to `data.max_length`, and pads each batch to its longest sequence.

GLUE test labels may be hidden and represented as -1. The collator omits such labels, so evaluation writes predictions without attempting a supervised metric. Local scoring uses the labeled validation partition. Prediction JSONL uses the text columns alone.

## Outputs

Training saves `history.jsonl` with one record per epoch, including train and validation losses and task metrics. Evaluation writes `metrics.json` and an `index`/`prediction` TSV. Image and classification predictions are integer class IDs. STS-B predictions are floating scores. These output types also apply when training from a local DatasetDict or ImageFolder.
