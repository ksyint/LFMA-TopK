"""Use the pretrained ViT processor on raw RGB images."""
import torch

class ImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, records):
        images, labels = zip(*records)
        batch = self.processor(images=[image.convert('RGB') for image in images], return_tensors='pt')
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        return batch
