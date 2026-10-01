# Experiment settings

Each Python or YAML file is a complete experiment, including its backbone, raw dataset, sparse support, scale, optimizer, seed, and output directory. Python profiles assign a literal dictionary to `cfg`. No external base configuration is needed to read a profile.

Profiles use this directory structure:

```text
experiments/
  vit-base-cifar10-ratio_0p0500-alpha_12-seed_42.yaml
  roberta-base/
    cola-ratio_0p0005-alpha_150-seed_1024.py
    mrpc/ratio_0p0016-alpha_150-seed_42.py
    cola/
      ratio_0p0016-alpha_150-seed_42.py
      ratio_0p0010/ratio_0p0010-alpha_150-seed_42.py
  vit-base/
    cifar10/
      ratio_0p1000-alpha_12-seed_42.yaml
      ratio_0p0500/
        ratio_0p0500-alpha_120-seed_42.yaml
        alpha_12/ratio_0p0500-alpha_12-seed_456.yaml
```

Representative profiles sit beside each backbone's task folders. CIFAR-10 separates support ratios and scales, while CoLA groups support ratios. Their task and ratio folders also retain representative profiles. The catalog separates optimization seeds from image split seeds. Every profile uses `data.split_seed: 42`, so varying `train.seed` does not change image train/validation/test membership.

| Protocol | Backbones | Support ratios | Scales | Profiles |
| --- | --- | --- | --- | --- |
| `table1` | ViT-B and ViT-L | 0.05 | 12 | 70 |
| `vision-sparse` | ViT-B and ViT-L | 0.0003 | 120 | 70 |
| `vision-ablation` | ViT-B and ViT-L | 0.0003, 0.05, 0.1 | 12, 120 | 420 |
| `table2` | RoBERTa-Base | 0.0016, 0.001, 0.0005 | 150 | 90 |
| `table3` | RoBERTa-Large | 0.0005, 0.0003, 0.0001 | 150 | 90 |
| `all` | All four | Backbone-specific | Backbone-specific | 600 |

Each image setting includes seven datasets. Each language setting includes SST-2, MRPC, QNLI, RTE, CoLA, and STS-B. The five optimization seeds are 42, 123, 456, 789, and 1024. Named protocols select settings from the same catalog, so overlapping selections share their experiment output directories.

Vision profiles use 50 epochs, batch size 32, AdamW coefficient/head learning rates of 1e-4, weight decay 1e-4, and constant learning rates. GLUE profiles use 30 epochs, batch size 32, coefficient learning rate 0.05, head learning rate 0.001, weight decay 1e-4, and scale 150. These values are explicit in each file and can be adjusted with `run.py train --opts`.

`run.py protocol` executes train then eval for each profile. Training selects `best/` by validation accuracy, CoLA Matthews correlation, or STS-B Pearson correlation. Image evaluation uses the untouched test partition. GLUE scoring uses its labeled validation partition. The generic `run.py grid` entry point trains only unless `--evaluate` is given.

```bash
python run.py protocol --protocol table1 --tasks cifar100 --backbones vit-large \
  --model-template '/weights/{backbone}' --data-root /datasets/vision
python run.py protocol --protocol table2 --tasks mrpc rte \
  --dataset-template '/datasets/glue-{task}' --model-template '/weights/{backbone}' --offline
```

Results follow the same task/support/scale/seed hierarchy under `--output-root`. Evaluation places `metrics.json` and `predictions.tsv` in each run's `evaluation/test/` or `evaluation/validation/` directory.

```bash
python run.py summarize --root results/catalog --output results/summary
```

`seed_records.json` records every completed run. `seed_summary.json` and `seed_summary.csv` group by backbone, task, support ratio, scale, split, and primary metric. They report the seed list, missing expected seeds, median, mean, sample standard deviation, minimum, and maximum. A duplicate seed within a group raises an error so that repeated copies of the same result do not count as independent runs. Metrics remain on their native 0–1 correlation/accuracy scale.

Use `--expected-seeds` when summarizing a deliberately smaller seed set. The summary never selects a support ratio or scale by test performance.

```bash
python run.py summarize --root results/catalog --expected-seeds 42 123 456 --output results/summary-three-seeds
python run.py grid --protocol vision-ablation --alphas 120 --seeds 42 --dry-run
```

The dry-run command checks every profile schema and output path, then reports the selected profiles. It does not load or download a model. `python run.py catalog` regenerates the committed catalog from the same axes and keeps each profile's configuration format.

`fit` records CUDA timing, peak memory, gradient norms, optimizer groups, and aligned validation logits. It keeps the constant AdamW schedule by default. Optional linear or cosine schedules use optimizer-update coordinates, including gradient accumulation. Resume restores the optimizer, precision state, loader generator, sampler, and random streams from an epoch checkpoint.

Each accumulation window averages over its actual number of examples, including a shorter final batch. `data-contract.json` records the ordered training and validation input fingerprints and loader sizes. Text fingerprints cover the task input fields. Image fingerprints cover decoded RGB pixels and dimensions. Resume verifies those fingerprints along with optimizer tensor names, shapes, group order, and update coordinates.

`last/` and `best/` each contain `adapter_model.safetensors`, `adapter_config.json`, the processor, `execution_state.pt`, `execution.json`, `validation.npz`, and a hashed file inventory. The complete checkpoint is prepared in a temporary directory before replacing its destination. The epoch journal is written first so that a resumed checkpoint can discard later incomplete work. Checkpoint loading recovers the preceding complete directory if a process stopped between directory replacements.

Set `train.patience` to a positive number to stop after that many validation epochs without an improvement exceeding `train.minimum_improvement`. The comparison anchor advances only after a qualifying improvement. `best/` still tracks every increase in the primary validation score. Both counters and the comparison anchor are restored when resuming. A run that already met its stopping criterion returns its saved history without starting another epoch.

```bash
python run.py fit --config config.yaml --opts train.save_dir results/detailed
python run.py fit --resume results/detailed/last
python run.py evaluate-detailed --checkpoint results/detailed/best --split test --output results/detailed/evaluation/test
python run.py analyze --predictions results/detailed/evaluation/test/predictions.npz --output results/detailed/analysis
```

Detailed evaluation saves per-class confusion and recall, top-k accuracy, reliability bins, Brier score, and confidence-based coverage. STS-B reports Pearson and rank correlation, residual bins, and absolute-error examples. Bootstrap resamples groups when a manifest provides `group`. Use `analyze --baseline other/predictions.npz` for an identity-aligned paired comparison. `calibrate --validation validation.npz --predictions test.npz --output results/calibrated` fits its temperature only from validation labels.

```bash
python run.py audit-data --manifest-dir datasets/cifar10 --inspect-images --output results/data-audit.json
python run.py subset --manifest-dir datasets/cifar10 --count 1000 --output datasets/cifar10-subset
python run.py budget --backbone vit-base --ratio 0.0003 --task cifar10
python run.py budget-grid --backbones vit-base vit-large --ratios 0.0003 0.05 0.1 --output results/budgets.json
python run.py diagnostics --checkpoint results/detailed/best --output results/spectrum.json --singular-values
```

The dataset audit checks content identities, conflicting labels, shared groups, image decoding, and label distribution. Subsets change only the training selection and preserve the evaluation partitions. `data.sampling.strategy=length` groups text examples by length during training, while evaluation always follows manifest order.

Length sampling uses the selected pretrained tokenizer with the same special tokens and truncation length as the collator. Complete batches are shuffled within the epoch and a partial final batch stays at the end. The sampler saves the token-length fingerprint, bucket size, and epoch for resume.

Prediction archives include input-content fingerprints and optional manifest groups. Calibration checks the actual input content across the fitting and assessment sets, even when they have different split names or record IDs. Confidence coverage includes every example tied at the selected threshold. Paired comparisons check aligned reference labels, groups, and content fingerprints before resampling.

`compose --checkpoints run-a/best run-b/best --weights 0.5 0.5 --output results/composed` combines task-compatible sparse updates on CUDA. Each input scale is absorbed into its coefficients before support union. `--maximum-coefficients` retains the strongest union entries and reports retained spectral energy. The output remains a standard adapter directory accepted by `evaluate-detailed` and `predict`.

Composition compares the frozen CUDA backbone parameters and buffers across all inputs. Its per-layer report includes the spatial residual of the combined update against the weighted inputs. The saved adapter metadata records each layer's actual support size and scale after composition.
