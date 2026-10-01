# Workflow guides

## Benchmarks

- [COLA text benchmark](benchmarks/cola.md)
- [MRPC text benchmark](benchmarks/text/mrpc.md)
- [QNLI text benchmark](benchmarks/text/qnli.md)
- [RTE text benchmark](benchmarks/text/rte.md)
- [SST2 text benchmark](benchmarks/text/sst2.md)
- [STSB text benchmark](benchmarks/text/stsb.md)
- [cifar10 image benchmark](benchmarks/cifar10.md)
- [cifar100 image benchmark](benchmarks/vision/cifar100.md)
- [eurosat image benchmark](benchmarks/vision/eurosat.md)
- [fgvc_aircraft image benchmark](benchmarks/vision/fgvc_aircraft.md)
- [oxford_pets image benchmark](benchmarks/vision/oxford_pets.md)
- [resisc45 image benchmark](benchmarks/vision/resisc45.md)
- [stanford_cars image benchmark](benchmarks/vision/stanford_cars.md)

## Overview

- [Dataset preparation](datasets.md)
- [Experiment settings](experiments.md)
- [Modules and checkpoints](implementation.md)
- [Pretrained model files](pretrained-models.md)

## Models

- [roberta-base pretrained artifacts](experiments/roberta-base.md)
- [roberta-large pretrained artifacts](experiments/roberta-large.md)
- [vit-base pretrained artifacts](experiments/vit-base.md)
- [vit-large pretrained artifacts](experiments/vit-large.md)

## Protocols

- [table1 experiment selection](experiments/table1.md)
- [table2 experiment selection](experiments/table2.md)
- [table3 experiment selection](experiments/table3.md)
- [vision-ablation experiment selection](experiments/vision-ablation.md)
- [vision-sparse experiment selection](experiments/vision-sparse.md)

## Workflows

- [Validate portable adapter state](workflows/adapter-integrity.md)
- [Transfer an adapter archive](workflows/adapter-transport.md)
- [Prepare local image benchmark bundles](workflows/jsonl-images.md)
- [Prepare local text benchmark bundles](workflows/jsonl-text.md)
- [Export a standard Transformers model](workflows/merged-export.md)
- [Offline execution](workflows/offline.md)
- [Create and continue experiment plans](workflows/plan-session.md)
- [Continue trained adapters](workflows/resume.md)
- [Compare repeated-seed benchmarks](workflows/seed-comparison.md)
- [Measure learned Fourier support](workflows/spectral-diagnostics.md)
