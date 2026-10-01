"""Resolve local artifacts, cache locations, and offline execution."""
import os


def apply_runtime_options(config, args):
    for argument, section, key in [('cache_dir', 'model', 'cache_dir'), ('model_dir', 'model', 'local_dir'),
                                  ('data_root', 'data', 'root'), ('dataset_dir', 'data', 'local_dir'),
                                  ('imagefolder', 'data', 'imagefolder')]:
        value = getattr(args, argument, None)
        if value is not None:
            config[section][key] = value
    if getattr(args, 'offline', False):
        config['model']['offline'] = True
    if config['model']['offline']:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    return config
