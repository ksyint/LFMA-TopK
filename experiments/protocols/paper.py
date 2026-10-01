"""Named subsets for table settings and support/scale ablations."""
from .models import BACKBONES

PROTOCOLS = {
    'table1': 'Both ViTs, seven image datasets, support 0.05, scale 12, five seeds',
    'vision-sparse': 'Both ViTs, seven image datasets, support 0.0003, scale 120, five seeds',
    'vision-ablation': 'Both ViTs, three supports, two scales, seven datasets, five seeds',
    'table2': 'RoBERTa-Base, six GLUE tasks, three supports, five seeds',
    'table3': 'RoBERTa-Large, six GLUE tasks, three supports, five seeds',
    'all': 'Every complete pretrained model/task/support/scale/seed profile',
}


def accepts(config, protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown experiment protocol: {protocol}')
    backbone = config['model']['backbone']
    vision = BACKBONES[backbone]['family'] == 'vit'
    ratio, alpha = config['adapter']['top_k_ratio'], config['adapter']['alpha']
    if protocol == 'table1':
        return vision and ratio == 0.05 and alpha == 12.0
    if protocol == 'vision-sparse':
        return vision and ratio == 0.0003 and alpha == 120.0
    if protocol == 'vision-ablation':
        return vision
    if protocol == 'table2':
        return backbone == 'roberta-base'
    if protocol == 'table3':
        return backbone == 'roberta-large'
    return True
