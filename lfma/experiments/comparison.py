"""Repeated-seed benchmark comparisons with explicit task and support coordinates."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics

import numpy as np

from lfma.models.fourier.core import GLUE_TASKS


COORDINATES = ('backbone', 'task', 'ratio', 'alpha')


def coordinate(row):
    return tuple(row[name] for name in COORDINATES)


def collect_runs(root):
    rows = []
    for metadata_path in sorted(Path(root).rglob('best/adapter_config.json')):
        directory = metadata_path.parent.parent
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('format') != 'lfma-pretrained-v1':
            continue
        config = metadata['config']
        task = config['data']['task']
        split = 'validation' if task in GLUE_TASKS else 'test'
        scores_path = directory / 'evaluation' / split / 'metrics.json'
        if not scores_path.is_file():
            continue
        scores = json.loads(scores_path.read_text())
        primary = GLUE_TASKS.get(task, (None, None, 'accuracy'))[2]
        if primary not in scores:
            raise ValueError(f'Missing primary metric {primary} in {scores_path}')
        value = float(scores[primary])
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite benchmark score at {scores_path}')
        row = {
            'backbone': config['model']['backbone'],
            'task': task,
            'ratio': config['adapter']['top_k_ratio'],
            'alpha': config['adapter']['alpha'],
            'seed': config['train']['seed'],
            'metric': primary,
            'score': value,
            'epoch': metadata['epoch'],
            'split': split,
            'directory': str(directory.resolve()),
            'revision': config['model']['revision'],
            'adapter_real_scalars': sum(
                2 * target['k'] for target in metadata['targets'].values()
            ),
        }
        rows.append(row)
    identities = [(coordinate(row), row['seed']) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Several scored runs share the same task, support, scale, and seed')
    return rows


def summarize_runs(rows, expected_seeds=None):
    groups = defaultdict(list)
    for row in rows:
        groups[coordinate(row)].append(row)
    summaries = []
    for key, runs in sorted(groups.items()):
        values = [row['score'] for row in runs]
        seeds = sorted(row['seed'] for row in runs)
        revisions = {row['revision'] for row in runs}
        if len(revisions) != 1:
            raise ValueError(f'Cannot aggregate different pretrained revisions at {key}')
        summary = dict(zip(COORDINATES, key))
        summary.update(
            metric=runs[0]['metric'],
            runs=len(runs),
            seeds=seeds,
            missing_seeds=sorted(set(expected_seeds or []) - set(seeds)),
            mean=statistics.mean(values),
            median=statistics.median(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            minimum=min(values),
            maximum=max(values),
            revision=next(iter(revisions)),
            adapter_real_scalars=runs[0]['adapter_real_scalars'],
        )
        summaries.append(summary)
    return summaries


def pair_runs(baseline, candidate):
    left = {(row['backbone'], row['task'], row['seed']): row for row in baseline}
    right = {(row['backbone'], row['task'], row['seed']): row for row in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate):
        raise ValueError(
            'Select one support/scale setting per backbone/task before paired comparison'
        )
    if left.keys() != right.keys():
        raise ValueError(
            'Paired comparison needs matching backbone, task, and seed coordinates'
        )
    result = []
    for key in sorted(left):
        before, after = left[key], right[key]
        if before['metric'] != after['metric'] or before['revision'] != after['revision']:
            raise ValueError('Paired metric and pretrained revision must match')
        result.append(
            {
                'backbone': key[0],
                'task': key[1],
                'seed': key[2],
                'metric': before['metric'],
                'baseline': before['score'],
                'candidate': after['score'],
                'difference': after['score'] - before['score'],
                'baseline_parameters': before['adapter_real_scalars'],
                'candidate_parameters': after['adapter_real_scalars'],
            }
        )
    return result


def bootstrap_difference(pairs, repeats=2000, seed=42, confidence=0.95):
    if repeats < 1 or not 0 < confidence < 1:
        raise ValueError('Bootstrap requires positive repeats and confidence in (0,1)')
    groups = defaultdict(list)
    for row in pairs:
        groups[(row['backbone'], row['task'])].append(row['difference'])
    generator = np.random.default_rng(seed)
    results = []
    for (backbone, task), differences in sorted(groups.items()):
        values = np.asarray(differences, dtype=np.float64)
        selections = generator.integers(0, len(values), size=(repeats, len(values)))
        means = values[selections].mean(axis=1)
        tail = (1 - confidence) / 2
        results.append(
            {
                'backbone': backbone,
                'task': task,
                'paired_seeds': len(values),
                'mean_difference': float(values.mean()),
                'lower': float(np.quantile(means, tail)),
                'upper': float(np.quantile(means, 1 - tail)),
                'confidence': confidence,
                'repeats': repeats,
            }
        )
    return results


def pareto_front(summaries):
    groups = defaultdict(list)
    for row in summaries:
        groups[(row['backbone'], row['task'])].append(row)
    output = []
    for _, settings in sorted(groups.items()):
        for current in settings:
            dominated = any(
                other['adapter_real_scalars'] <= current['adapter_real_scalars']
                and other['mean'] >= current['mean']
                and (
                    other['adapter_real_scalars'] < current['adapter_real_scalars']
                    or other['mean'] > current['mean']
                )
                for other in settings
            )
            output.append(dict(current, pareto=not dominated))
    return output


def filter_runs(rows, ratio=None, alpha=None, tasks=None):
    return [
        row
        for row in rows
        if (ratio is None or row['ratio'] == ratio)
        and (alpha is None or row['alpha'] == alpha)
        and (tasks is None or row['task'] in tasks)
    ]


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def export_comparison(rows, output, expected_seeds=None, baseline=None, repeats=2000, seed=42):
    if not rows:
        raise ValueError('No completed benchmark runs were selected')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = summarize_runs(rows, expected_seeds)
    report = {
        'format': 'lfma-benchmark-comparison-v1',
        'runs': rows,
        'settings': summaries,
        'pareto': pareto_front(summaries),
    }
    write_csv(output / 'runs.csv', rows)
    write_csv(output / 'settings.csv', summaries)
    write_csv(output / 'pareto.csv', report['pareto'])
    if baseline is not None:
        pairs = pair_runs(baseline, rows)
        report['paired_intervals'] = bootstrap_difference(pairs, repeats, seed)
        write_csv(output / 'paired.csv', pairs)
        write_csv(output / 'paired_intervals.csv', report['paired_intervals'])
    (output / 'comparison.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n'
    )
    return report


def compare_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ratio', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--expected-seeds', type=int, nargs='+')
    parser.add_argument('--baseline-root')
    parser.add_argument('--baseline-ratio', type=float)
    parser.add_argument('--baseline-alpha', type=float)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = filter_runs(collect_runs(args.root), args.ratio, args.alpha, args.tasks)
    baseline = None
    if args.baseline_root:
        baseline = filter_runs(
            collect_runs(args.baseline_root),
            args.baseline_ratio,
            args.baseline_alpha,
            args.tasks,
        )
    result = export_comparison(
        rows, args.output, args.expected_seeds, baseline, args.bootstrap, args.seed
    )
    print(
        json.dumps({'runs': len(result['runs']), 'settings': len(result['settings'])}, indent=2)
    )
