"""One simulator/evaluation loop for all RoboCasa24 policies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from evaluation.protocol import (ROBOCASA_COMMIT, ROBOSUITE_COMMIT, validate_actions,
                                 EVALUATION_SEED_BLOCK, TASK_SEED_STRIDE, benchmark_defaults,
                                 add_benchmark_arguments, resolve_evaluation_seeds)

ROOT = Path(__file__).resolve().parents[1]


def check_runtime():
    import robocasa
    import robosuite
    def head(path):
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if Path(robocasa.__file__).resolve().parent != ROOT / 'robocasa':
        raise RuntimeError(f'Wrong RoboCasa import: {robocasa.__file__}')
    suite = ROOT / 'third_party/robosuite'
    if Path(robosuite.__file__).resolve().parent != suite / 'robosuite':
        raise RuntimeError(f'Install the pinned submodule, imported {robosuite.__file__}')
    if head(suite) != ROBOSUITE_COMMIT:
        raise RuntimeError('RoboSuite revision mismatch')
    if subprocess.check_output(['git', '-C', str(suite), 'diff', 'HEAD', '--', 'robosuite'], text=True):
        raise RuntimeError('RoboSuite sources have uncommitted modifications')
    # HEAD may advance after evaluation code is committed. Check simulation sources separately.
    diff = subprocess.check_output(['git', '-C', str(ROOT), 'diff', ROBOCASA_COMMIT,
                                    '--', 'robocasa'], text=True)
    if diff:
        raise RuntimeError('RoboCasa simulation sources differ from the pinned revision')
    return {'robocasa_head': head(ROOT), 'robosuite_head': head(suite)}


def rollout(env, policy, task, episode_id, seed, horizon, replan_steps, video_path=None, action_bounds="error", video_stride=4):
    from evaluation.environment import base_transform, observe
    import imageio.v2 as imageio
    if video_stride <= 0:
        raise ValueError('video_stride must be positive')
    if action_bounds not in ('error', 'controller'):
        raise ValueError('action_bounds must be error or controller')
    out_of_bounds_components, max_bound_violation = 0, 0.
    episode_started = time.perf_counter()
    timings = dict(reset_seconds=0., observation_seconds=0., policy_roundtrip_seconds=0.,
                   env_step_seconds=0., success_check_seconds=0., video_render_seconds=0.,
                   video_write_seconds=0., video_close_seconds=0.)
    tick = time.perf_counter()
    env.reset()
    timings['reset_seconds'] = time.perf_counter() - tick
    base = base_transform(env)
    initial_state = np.asarray(env.sim.get_state().flatten())
    policy.reset({'task': task, 'episode_id': episode_id, 'seed': seed})
    step, success, calls, inference_seconds = 0, False, 0, 0.
    inference_requests = []
    video_frames = 0
    writer = imageio.get_writer(str(video_path), fps=env.control_freq / video_stride) if video_path else None
    try:
        while step < horizon:
            tick = time.perf_counter()
            obs = observe(env, base, task, episode_id, step)
            timings['observation_seconds'] += time.perf_counter() - tick
            started = time.perf_counter()
            request_started_unix = time.time()
            chunk = validate_actions(policy.predict(obs))
            request_seconds = time.perf_counter() - started
            inference_seconds += request_seconds
            inference_requests.append(dict(step=step, seconds=request_seconds, started_unix_seconds=request_started_unix))
            timings['policy_roundtrip_seconds'] += request_seconds
            calls += 1
            if len(chunk) < replan_steps:
                raise ValueError(f'Policy returned {len(chunk)} actions, fewer than common K={replan_steps}')
            for action in chunk[:min(replan_steps, horizon - step)]:
                low, high = env.action_spec
                violation = np.maximum(low[:7] - action, action - high[:7])
                outside = violation > 1e-6
                out_of_bounds_components += int(outside.sum())
                max_bound_violation = max(max_bound_violation, float(violation.max()))
                if outside.any() and action_bounds == 'error':
                    raise ValueError('Native action outside controller bounds; choose controller mode explicitly to delegate saturation')
                padded = np.zeros_like(low)
                padded[:7] = action
                tick = time.perf_counter()
                env.step(padded)
                timings['env_step_seconds'] += time.perf_counter() - tick
                step += 1
                if writer and (step - 1) % video_stride == 0:
                    tick = time.perf_counter()
                    frame = env.sim.render(height=256, width=256, camera_name='robot0_agentview_left')
                    timings['video_render_seconds'] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    writer.append_data(frame[::-1])
                    video_frames += 1
                    timings['video_write_seconds'] += time.perf_counter() - tick
                tick = time.perf_counter()
                succeeded = env._check_success()
                timings['success_check_seconds'] += time.perf_counter() - tick
                if succeeded:
                    success = True
                    break
            if success:
                break
    finally:
        if writer:
            tick = time.perf_counter()
            writer.close()
            timings['video_close_seconds'] += time.perf_counter() - tick
    return dict(seed=seed, episode_id=episode_id, success=success, steps=step,
                inference_calls=calls, inference_seconds=inference_seconds,
                video_stride=video_stride, video_frames=video_frames,
                video_fps=env.control_freq / video_stride if video_path else None,
                action_bounds=action_bounds, out_of_bounds_components=out_of_bounds_components,
                max_bound_violation=max_bound_violation, timings=timings, episode_wall_seconds=time.perf_counter() - episode_started,
                inference_requests=inference_requests,
                initial_state_sha256=hashlib.sha256(initial_state.tobytes()).hexdigest())


def run_episode_job(job):
    """Spawn-safe worker; no model weights and no shared simulator state."""
    from evaluation.transport import RemotePolicy
    from evaluation.environment import create_env, render_backend
    policy = RemotePolicy(job['endpoint'], job['timeout_ms'])
    env = None
    try:
        tick = time.perf_counter()
        env = create_env(job['task'], job['seed'])
        create_seconds = time.perf_counter() - tick
        record = rollout(env, policy, job['task'], job['seed'], job['seed'], job['horizon'],
                         job['replan_steps'], Path(job['video']) if job['video'] else None,
                         job['action_bounds'], job['video_stride'])
        record.update(evaluation_seed=job['evaluation_seed'], trial_index=job['trial_index'])
        record['render_backend'] = render_backend(env)
        tick = time.perf_counter()
        env.close()
        env = None
        record['timings'].update(create_env_seconds=create_seconds,
                                 close_env_seconds=time.perf_counter() - tick)
        return job['task'], record
    finally:
        if env is not None:
            env.close()
        policy.close()


def episode_seed(task, episode_index, evaluation_seed, registry):
    if evaluation_seed < 0 or not 0 <= episode_index < TASK_SEED_STRIDE:
        raise ValueError('Invalid evaluation seed or trial index')
    seed = evaluation_seed * EVALUATION_SEED_BLOCK + list(registry).index(task) * TASK_SEED_STRIDE + episode_index
    if not 0 <= seed < 2**32:
        raise ValueError('Generated environment seed exceeds uint32 range')
    return seed


def episode_specs(tasks, episodes, evaluation_seeds, registry, shard_index=0, shard_count=1):
    if not 0 < episodes <= TASK_SEED_STRIDE or shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError('Invalid episode count or shard index/count')
    seeds = [evaluation_seeds] if isinstance(evaluation_seeds, int) else list(evaluation_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('Evaluation seeds must be distinct and nonnegative')
    if len(registry) * TASK_SEED_STRIDE > EVALUATION_SEED_BLOCK:
        raise ValueError('Task registry exceeds evaluation seed block')
    result = []
    index = 0
    for evaluation_seed in seeds:
        for task in tasks:
            for i in range(episodes):
                seed = episode_seed(task, i, evaluation_seed, registry)
                if index % shard_count == shard_index:
                    result.append(dict(task=task, seed=seed, evaluation_seed=evaluation_seed,
                                       trial_index=i, horizon=registry[task]))
                index += 1
    return result


def success_metrics(by_task):
    per_task = {task: sum(r['success'] for r in records) / len(records)
                for task, records in by_task.items() if records}
    grouped = {}
    for task, records in by_task.items():
        for record in records:
            seed = str(record.get('evaluation_seed', 0))  # historical single-seed records
            grouped.setdefault(seed, {}).setdefault(task, []).append(record)
    per_seed = {}
    for seed, tasks in grouped.items():
        rates = {task: sum(r['success'] for r in rows) / len(rows) for task, rows in tasks.items()}
        per_seed[seed] = dict(per_task=rates, macro_success_rate=float(np.mean(list(rates.values()))),
                              completed_episodes=sum(map(len, tasks.values())))
    values = [row['macro_success_rate'] for row in per_seed.values()]
    return dict(per_task=per_task, macro_success_rate=float(np.mean(list(per_task.values()))),
                per_evaluation_seed=per_seed,
                mean_success_rate_across_seeds=float(np.mean(values)),
                std_success_rate_across_seeds=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                seed_std_ddof=1)


def episode_plan(tasks, episodes, seeds, specs):
    return dict(benchmark=benchmark_defaults()['name'], tasks=tasks, evaluation_seeds=seeds,
                trials_per_task_per_seed=episodes, total_episodes=len(tasks) * episodes * len(seeds),
                assigned_episodes=len(specs),
                episodes_per_evaluation_seed={str(s): len(tasks) * episodes for s in seeds})


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='tcp://127.0.0.1:9100')
    add_benchmark_arguments(parser)
    parser.add_argument('--timeout-ms', type=int, default=120000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--video-stride', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--action-bounds', choices=['error', 'controller'], default='error')
    args = parser.parse_args()
    registry = json.loads((ROOT / 'evaluation/configs/tasks.json').read_text())
    tasks = list(registry) if args.tasks == ['all'] else args.tasks
    if min(args.episodes, args.replan_steps, args.video_stride, args.workers) <= 0 or len(set(tasks)) != len(tasks):
        parser.error('episodes/K/video-stride/workers must be positive and tasks distinct')
    if any(task not in registry for task in tasks):
        parser.error('Unknown task')
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error('Invalid shard index/count')
    try:
        args.seeds = resolve_evaluation_seeds(args)
        specs = episode_specs(tasks, args.episodes, args.seeds, registry, args.shard_index, args.shard_count)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(episode_plan(tasks, args.episodes, args.seeds, specs), indent=2))
        return
    if not specs:
        parser.error('No episodes assigned to this shard')
    revisions = check_runtime()
    from evaluation.transport import RemotePolicy
    probe = RemotePolicy(args.endpoint, args.timeout_ms)
    try:
        metadata = probe.metadata
    finally:
        probe.close()
    if args.workers > 1 and not metadata.get('parallel_safe', False):
        parser.error('Shared server has not declared episode isolation; use one worker')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    revisions=revisions, policy=metadata, horizons=registry, benchmark_protocol=benchmark_defaults(),
                    rendering_environment={k: os.environ.get(k) for k in
                        ['MUJOCO_GL', 'MUJOCO_EGL_DEVICE_ID', '__EGL_VENDOR_LIBRARY_FILENAMES']})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    assigned_tasks = [task for task in tasks if any(j['task'] == task for j in specs)]
    records = {task: [] for task in assigned_tasks}
    expected = {task: sum(j['task'] == task for j in specs) for task in assigned_tasks}
    for task in assigned_tasks:
        (args.output / task).mkdir()
    jobs = [dict(**spec, endpoint=args.endpoint, timeout_ms=args.timeout_ms,
                 replan_steps=args.replan_steps, action_bounds=args.action_bounds,
                 video_stride=args.video_stride,
                 video=str(args.output / spec['task'] / f"{spec['seed']}.mp4") if args.video else None)
            for spec in specs]
    started = time.perf_counter()
    def save_result(result):
        task, record = result
        records[task].append(record)
        records[task].sort(key=lambda r: r['seed'])
        (args.output / task / 'episodes.json').write_text(json.dumps(records[task], indent=2))
        summary = dict(**success_metrics(records), evaluation_seeds=args.seeds,
                       trials_per_task_per_seed=args.episodes,
                       completed_tasks=sum(len(rows) == expected[task] for task, rows in records.items()),
                       requested_tasks=len(assigned_tasks), completed_episodes=sum(map(len, records.values())),
                       requested_episodes=len(jobs), workers=args.workers,
                       run_wall_seconds=time.perf_counter() - started)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(task, 'seed', record['seed'], 'success', record['success'], 'steps', record['steps'], flush=True)
    if args.workers == 1:
        for job in jobs:
            save_result(run_episode_job(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as executor:
            futures = [executor.submit(run_episode_job, job) for job in jobs]
            try:
                for future in as_completed(futures):
                    save_result(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    summary = json.loads((args.output / 'summary.json').read_text())
    summary.update(run_wall_seconds=time.perf_counter() - started, complete=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
