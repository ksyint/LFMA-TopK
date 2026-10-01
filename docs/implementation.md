# Modules and checkpoints

`run.py` selects preparation, training, evaluation, prediction, and experiment protocols. Model construction and configuration live in `lfma/models/fourier/core.py`, with sparse update operations in its `adaptation` package.

| Module | Responsibility |
| --- | --- |
| `lfma/models/fourier/core.py` | Pinned model and dataset identities, projection selection, and configuration |
| `lfma/models/fourier/spectrum.py` | CUDA support measurements and spectral reports |
| `lfma/models/fourier/adaptation/layers.py` | Stable support selection, sparse coefficients, and the linear update |
| `lfma/models/fourier/adaptation/injection.py` | Named projection replacement and merged model construction |
| `lfma/data/benchmarks/streams.py` | Raw datasets, fixed partitions, processors, collators, and task metrics |
| `lfma/artifacts/adapter/storage.py` | Sparse tensor serialization, strict restoration, merged artifacts, and seed summaries |
| `lfma/artifacts/adapter/validation.py` | Tensor header, processor, and configuration checks |
| `lfma/artifacts/adapter/bundle.py` | Verified adapter packing and extraction |
| `lfma/experiments/protocols.py` | Profile catalogs, named protocols, run selection, and parameter inspection |
| `lfma/experiments/session.py` | Resumable training and evaluation plans |
| `lfma/experiments/comparison.py` | Repeated-seed aggregation and paired comparisons |
| `run.py` | CUDA optimization, model preparation, evaluation, prediction, and command dispatch |
| `experiments/configs/catalog/` | Complete vision and GLUE experiment configurations |

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

`run.py inspect` computes the configured encoder and head sizes without loading weights. When given `--checkpoint`, it checks coefficient and classifier counts from the safetensors header. Support-index buffers are reported separately from trainable scalars.

Merged exports replace every LFMA wrapper with a standard linear projection containing the adapted weight. The exported Hugging Face model includes the learned head and processor. Its `lfma_merge.json` manifest records the base identity, revision, task, support configuration, selected epoch, target projections, and file sizes.
