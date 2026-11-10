"""Summarize shared-client timings; policy RTT includes serialization, broker and inference."""
import argparse
import json
from pathlib import Path
import numpy as np


def distribution(values):
    return dict(count=len(values), mean_seconds=float(np.mean(values)),
                p50_seconds=float(np.percentile(values, 50)), p95_seconds=float(np.percentile(values, 95))) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    records = []
    for path in sorted(args.run.glob('*/episodes.json')):
        records.extend(json.loads(path.read_text()))
    if not records:
        raise ValueError('No completed episode records')
    totals = {}
    for record in records:
        for phase, seconds in record['timings'].items():
            totals[phase] = totals.get(phase, 0.) + seconds
    wall = sum(r['episode_wall_seconds'] + r['timings']['create_env_seconds'] +
               r['timings']['close_env_seconds'] for r in records)
    request_records = [q for r in records for q in r['inference_requests']]
    if all('started_unix_seconds' in q for q in request_records):
        request_records.sort(key=lambda q: q['started_unix_seconds'])
    requests = [q['seconds'] for q in request_records]
    steps = sum(r['steps'] for r in records)
    result = dict(episodes=len(records), success_count=sum(r['success'] for r in records),
                  steps=steps, control_hz=20, simulated_seconds=steps / 20,
                  measured_episode_wall_seconds=wall, simulated_seconds_per_wall_second=(steps / 20) / wall,
                  totals_seconds=totals, phase_percent={k: 100 * v / wall for k, v in totals.items()},
                  unaccounted_seconds=wall - sum(totals.values()),
                  first_policy_request_seconds=requests[0],
                  subsequent_policy_requests=distribution(requests[1:]),
                  env_step_ms_per_control_step=1000 * totals['env_step_seconds'] / steps,
                  observation_ms_per_request=1000 * totals['observation_seconds'] / len(requests))
    summary_path = args.run / 'summary.json'
    run_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    run_wall = run_summary.get('run_wall_seconds', wall)
    result['simulated_seconds_per_wall_second'] = (steps / 20) / run_wall
    result.update(run_wall_seconds=run_wall, summed_episode_wall_seconds=wall,
                  workers=run_summary.get('workers', 1),
                  actual_simulated_seconds_per_wall_second=(steps / 20) / run_wall,
                  phase_percentage_basis='summed worker times; overlapping phases are not a wall-time decomposition')
    (args.run / 'timing_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
