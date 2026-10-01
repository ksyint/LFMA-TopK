"""ImageNet-21k ViT encoder, task head, and saved image preprocessing."""
import torch


def load(origin, options, classes, processor_dir=None):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(
        origin, num_labels=classes, ignore_mismatched_sizes=True,
        attn_implementation='eager', torch_dtype=torch.float32, **options)
    processor_options = {'local_files_only': True} if processor_dir else dict(options)
    processor = AutoImageProcessor.from_pretrained(str(processor_dir or origin),
                                                   use_fast=False, **processor_options)
    return model, processor
