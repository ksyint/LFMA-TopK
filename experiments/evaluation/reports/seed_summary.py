"""Aggregate completed seed runs without selecting settings on held-out test scores."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from experiments.protocols.datasets import GLUE_TASKS


def collect_results(root, split='auto'):
    records = []
    for path in sorted(Path(root).rglob('best/adapter_config.json')):
        metadata = json.loads(path.read_text())
        config = metadata['config']
        task, run = config['data']['task'], path.parent.parent
        partition = ('validation' if task in GLUE_TASKS else 'test') if split == 'auto' else split
        primary = GLUE_TASKS.get(task, (None, None, 'accuracy'))[2]
        metrics_path = run / 'evaluation' / partition / 'metrics.json'
        if not metrics_path.is_file():
            continue
        scores = json.loads(metrics_path.read_text())
        if primary not in scores:
            continue
        records.append({'backbone': config['model']['backbone'], 'task': task,
                        'ratio': config['adapter']['top_k_ratio'], 'alpha': config['adapter']['alpha'],
                        'seed': config['train']['seed'], 'split': partition, 'metric': primary,
                        'score': scores[primary], 'selected_epoch': metadata['epoch'],
                        'scores': scores, 'run': str(run)})
    return records


def summarize_results(records, expected_seeds=(42, 123, 456, 789, 1024)):
    groups = defaultdict(list)
    keys = ('backbone', 'task', 'ratio', 'alpha', 'split', 'metric')
    for record in records:
        groups[tuple(record[key] for key in keys)].append(record)
    summary = []
    for key, entries in sorted(groups.items()):
        seeds = [entry['seed'] for entry in entries]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f'Duplicate seed results for {key}')
        scores = [entry['score'] for entry in entries]
        result = dict(zip(keys, key))
        result.update(runs=len(entries), seeds=sorted(seeds),
                      missing_seeds=sorted(set(expected_seeds) - set(seeds)),
                      median=statistics.median(scores), mean=statistics.mean(scores),
                      std=statistics.stdev(scores) if len(scores) > 1 else 0.0,
                      minimum=min(scores), maximum=max(scores))
        summary.append(result)
    return summary


def write_summary(records, output, expected_seeds=(42, 123, 456, 789, 1024)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(records, expected_seeds)
    (output / 'seed_records.json').write_text(json.dumps(records, indent=2) + '\n')
    (output / 'seed_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fields = ['backbone', 'task', 'ratio', 'alpha', 'split', 'metric', 'runs', 'seeds',
              'missing_seeds', 'median', 'mean', 'std', 'minimum', 'maximum']
    with (output / 'seed_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in summary:
            writer.writerow({name: json.dumps(value) if isinstance(value, list) else value for name, value in result.items()})
    return summary
