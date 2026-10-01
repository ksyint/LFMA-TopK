"""Task-shaped text pairs, padding, and regression/classification labels."""
import torch

from experiments.tasks.definitions import GLUE_TASKS

class TextCollator:
    def __init__(self, tokenizer, config):
        self.tokenizer, self.config = tokenizer, config
        self.first, self.second, _ = GLUE_TASKS[config['data']['task']]

    def __call__(self, records):
        first = [record[self.first] for record in records]
        second = [record[self.second] for record in records] if self.second else None
        batch = self.tokenizer(first, second, truncation=True, padding=True,
                               max_length=self.config['data']['max_length'], return_tensors='pt')
        if all('label' in record and record['label'] >= 0 for record in records):
            dtype = torch.float32 if self.config['data']['task'] == 'stsb' else torch.long
            batch['labels'] = torch.tensor([record['label'] for record in records], dtype=dtype)
        return batch
