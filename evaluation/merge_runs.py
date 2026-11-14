"""Validate and combine completed, nonoverlapping evaluation seed runs."""
import argparse
import csv
import json
from pathlib import Path

from evaluation.client import ROOT, episode_specs, success_metrics
from evaluation.protocol import benchmark_defaults


def merge_runs(sources, output, seeds):
    manifests = [json.loads((source / 'manifest.json').read_text()) for source in sources]
    summaries = [json.loads((source / 'summary.json').read_text()) for source in sources]
    reference = manifests[0]
    for manifest, summary in zip(manifests, summaries):
        if not summary.get('complete'):
            raise ValueError('Source evaluation is incomplete')
        for key in ('backend', 'checkpoint', 'config', 'revisions', 'tasks',
                    'trials_per_task_per_seed', 'replan_steps'):
            if manifest[key] != reference[key]:
                raise ValueError(f'Source runs have different {key}')
        for key in ('name', 'environment_seed_scheme', 'evaluation_seed_block', 'task_seed_stride'):
            if manifest['benchmark_protocol'][key] != reference['benchmark_protocol'][key]:
                raise ValueError(f'Source seed protocols differ: {key}')
    registry = json.loads((ROOT / 'evaluation/configs/tasks.json').read_text())
    trials = reference['trials_per_task_per_seed']
    expected = {(r['task'], r['seed']): r for r in episode_specs(reference['tasks'], trials, seeds, registry)}
    seen = set()
    by_task = {task: [] for task in reference['tasks']}
    for source, manifest, summary in zip(sources, manifests, summaries):
        source_expected = {(r['task'], r['seed']) for r in
                           episode_specs(manifest['tasks'], trials, manifest['evaluation_seeds'], registry)}
        source_seen = set()
        for task in reference['tasks']:
            for row in json.loads((source / task / 'episodes.json').read_text()):
                key = (task, row['seed'])
                if key in seen or key not in expected or key not in source_expected:
                    raise ValueError(f'Duplicate or unexpected episode: {key}')
                spec = expected[key]
                if any(row[name] != spec[name] for name in ('evaluation_seed', 'trial_index')):
                    raise ValueError(f'Mislabeled evaluation seed/trial: {key}')
                seen.add(key)
                source_seen.add(key)
                by_task[task].append(row)
        if source_seen != source_expected or len(source_seen) != summary['completed_episodes']:
            raise ValueError(f'Source episode coverage mismatch: {source}')
    if seen != set(expected):
        raise ValueError(f'Missing {len(set(expected) - seen)} episodes')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(reference, evaluation_seeds=seeds, requested_episodes=len(expected),
                    benchmark_protocol=benchmark_defaults(), source_runs=[str(source) for source in sources],
                    aggregation='Preserved original records; combined disjoint evaluation seed runs')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for task, rows in by_task.items():
        rows.sort(key=lambda row: row['seed'])
        (output / task).mkdir()
        (output / task / 'episodes.json').write_text(json.dumps(rows, indent=2))
    records = [row for rows in by_task.values() for row in rows]
    evaluation_seconds = sum(s['run_wall_seconds'] for s in summaries)
    summary = dict(complete=True, **success_metrics(by_task),
                   completed_tasks=len(by_task), requested_tasks=len(by_task),
                   completed_episodes=len(seen), requested_episodes=len(expected),
                   success_count=sum(row['success'] for row in records),
                   evaluation_seeds=seeds, trials_per_task_per_seed=trials,
                   steps=sum(row['steps'] for row in records),
                   run_wall_seconds=evaluation_seconds,
                   total_wall_seconds=sum(s['total_wall_seconds'] for s in summaries),
                   startup_and_warmup_seconds=sum(s['startup_and_warmup_seconds'] for s in summaries),
                   episodes_per_minute=60 * len(seen) / evaluation_seconds,
                   replicas=len(reference['replicas']), workers_per_gpu=reference['workers_per_gpu'],
                   max_workers_per_task=reference['max_workers_per_task'],
                   timing_basis='Sum of the sequential source runs; includes one startup per source run',
                   source_runs=[str(source) for source in sources])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'progress.json').write_text(json.dumps(summary, indent=2))
    with (output / 'task_success_rates.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task', 'success_count', 'episodes', 'success_rate_percent',
                         *[f'seed{seed}_percent' for seed in seeds]])
        for task, rows in by_task.items():
            writer.writerow([task, sum(row['success'] for row in rows), len(rows),
                             100 * summary['per_task'][task],
                             *[100 * summary['per_evaluation_seed'][str(seed)]['per_task'][task] for seed in seeds]])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=benchmark_defaults()['evaluation_seeds'])
    args = parser.parse_args()
    result = merge_runs([source.resolve() for source in args.runs], args.output.resolve(), args.seeds)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
