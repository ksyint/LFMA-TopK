#!/usr/bin/env python3
"""Export per-task medians, means, and sample deviations across completed seeds."""
import argparse

from experiments.evaluation.reports import collect_results, write_summary


def main(args):
    records = collect_results(args.root, args.split)
    if not records:
        raise ValueError('No scored runs found. Run eval.py with --output <run>/evaluation/<split> first.')
    summary = write_summary(records, args.output, args.expected_seeds)
    print(f'Wrote {len(summary)} task/setting summaries from {len(records)} evaluated runs to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/catalog')
    parser.add_argument('--split', choices=['auto', 'validation', 'test'], default='auto')
    parser.add_argument('--expected-seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    parser.add_argument('--output', default='results/summary')
    main(parser.parse_args())
