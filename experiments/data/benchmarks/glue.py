"""GLUE text fields, supervised labels, and dynamic tokenizer padding."""

from experiments.tasks.definitions import GLUE_TASKS, GLUE_REVISION
from .hub import dataset_from_hub
from ..preprocessing import TextCollator


def read_glue(config, split):
    task = config['data']['task']
    return dataset_from_hub(config, 'nyu-mll/glue', task, GLUE_REVISION, split)
