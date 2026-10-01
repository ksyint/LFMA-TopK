# Modules and checkpoints

The root scripts expose individual experiment operations. `train.py` handles configuration, checkpoints, and epoch selection. The transformer and data details stay in their own packages.

| Module | Responsibility |
| --- | --- |
| `experiments/protocols/models.py` | Pinned encoder identities and architecture sizes |
| `experiments/protocols/datasets/` | Dataset names, label counts, task fields, and pinned Hub revisions |
| `experiments/tasks/models/vision/` | ImageNet-21k ViT construction and image processor |
| `experiments/tasks/models/language/` | RoBERTa construction and tokenizer |
| `experiments/tasks/models/projections.py` | Query or query/value target selection |
| `experiments/data/benchmarks/` | Raw datasets, fixed partitions, and DataLoaders |
| `experiments/data/preprocessing/` | Batch image processing and dynamic text padding |
| `experiments/optimization/loops/` | Supervised forward pass, gradient accumulation, and prediction |
| `adapters/io/pretrained/` | Sparse tensor serialization, model identity, and restoration |
| `experiments/evaluation/reports/` | Parameter budgets and repeated-seed aggregation |

The frozen projection weight has shape `[out, in]`. Support selection computes a two-dimensional FFT of the initialization probe and retains the largest `floor(out * in * ratio)` magnitudes. The selected indices remain fixed. Each selected complex coefficient is stored as two FP32 trainable values. The real inverse FFT, multiplied by `alpha`, produces the update applied in the linear layer.

ViT receives a query adapter in every encoder block. RoBERTa receives query and value adapters in every block. Other encoder parameters are frozen. The task classifier is trainable and has its own learning rate. STS-B uses the same RoBERTa classification module with a single regression output.

The optimizer has a coefficient group and a task-head group. Gradient accumulation divides each microbatch loss by the number of microbatches in its current window, including the last partial window. Gradient clipping runs immediately before each AdamW step. FP32 is the default, and BF16 autocasting can be enabled for the transformer operations.

An adapter checkpoint is a directory:

```text
best/
  adapter_config.json
  adapter_model.safetensors
  processor/
  training_state.pt
```

`adapter_model.safetensors` contains coefficients, support indices, and the full learned classifier. The base transformer is identified by its pinned model revision rather than copied into every adapter. Restoration loads that encoder, checks target projection names and tensor shapes, attaches the saved supports, and loads the classifier strictly.

`adapter_config.json` records the full experiment configuration, target scales and support sizes, selected epoch, and best validation score. `processor/` stores the image processor or tokenizer used in the run. `training_state.pt` stores AdamW state and Torch/CUDA random states. Resume restores these states, then applies the requested coefficient/head learning rates and weight decay.

`inspect_adapter.py` computes the configured encoder and head sizes without loading weights. When given `--checkpoint`, it checks coefficient and classifier counts from the safetensors header. Support-index buffers are reported separately from trainable scalars.

Merged exports replace every LFMA wrapper with a standard linear projection containing the adapted weight. The exported Hugging Face model includes the learned head and processor. Its `lfma_merge.json` manifest records the base identity, revision, task, support configuration, selected epoch, target projections, and file sizes.
