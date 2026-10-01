"""RoBERTa classification/regression head and pretrained tokenizer."""
import torch


def load(origin, options, classes, processor_dir=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        origin, num_labels=classes, ignore_mismatched_sizes=True,
        attn_implementation='eager', torch_dtype=torch.float32, **options)
    tokenizer_options = {'local_files_only': True} if processor_dir else dict(options)
    tokenizer = AutoTokenizer.from_pretrained(str(processor_dir or origin), **tokenizer_options)
    return model, tokenizer
