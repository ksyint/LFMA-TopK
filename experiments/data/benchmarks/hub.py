"""Hub dataset caching with a portable save_to_disk override."""
import os
from pathlib import Path


def dataset_from_hub(config, repo_id, subset=None, revision=None, split=None, data_files=None):
    if config['model']['offline']:
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
    from datasets import DownloadConfig, load_dataset, load_from_disk
    data = config['data']
    if data['local_dir']:
        path = Path(data['local_dir'])
        if not path.is_dir():
            raise FileNotFoundError(f'Local dataset directory does not exist: {path}')
        result = load_from_disk(str(path))
        return result[split] if split else result
    return load_dataset(repo_id, subset, revision=revision, split=split, data_files=data_files,
                        cache_dir=data['cache_dir'],
                        download_config=DownloadConfig(local_files_only=config['model']['offline']))
