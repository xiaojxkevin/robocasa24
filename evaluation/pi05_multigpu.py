"""Managed pi0/pi05/Spatial Forcing replicas; each task belongs to one GPU and at most eight workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

import numpy as np

from evaluation.client import ROOT, check_runtime, episode_specs
from evaluation.multigpu import Processes, gpu_inventory, merge_results
from evaluation.protocol import (CAMERAS, OPENPI_NATIVE_CONTRACTS, add_benchmark_arguments,
                                 benchmark_defaults, resolve_evaluation_seeds, validate_openpi_config)
from evaluation.transport import RemotePolicy


def assign_tasks(tasks, registry, count):
    """Greedy horizon balancing, retaining unique task ownership."""
    groups = [[] for _ in range(count)]
    loads = [0] * count
    for task in sorted(tasks, key=lambda task: -registry[task]):
        rank = min(range(count), key=lambda rank: loads[rank])
        groups[rank].append(task)
        loads[rank] += registry[task]
    return groups


def cpu_partitions(count):
    """Keep XLA from creating one whole-machine thread pool per replica."""
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 2 * count:
        raise ValueError('Need at least two available CPU cores per GPU')
    result = []
    for rank in range(count):
        group = available[rank::count]
        model_count = max(1, len(group) // 3)
        result.append((group[:model_count], group[model_count:]))
    return result


def warmup_replica(replica, timeout_ms):
    policy = RemotePolicy(replica['endpoint'], timeout_ms)
    try:
        episode = dict(task='__warmup__', episode_id=-1, seed=0)
        policy.reset(episode)
        obs = dict(**episode, step_id=0, instruction='close the cabinet doors',
                   state10=np.array([.3, 0, .5, 1, 0, 0, 0, 1, 0, 1], np.float32),
                   images={camera: np.zeros((256, 256, 3), np.uint8) for camera in CAMERAS})
        started = time.perf_counter()
        actions = policy.predict(obs)
        return dict(gpu=replica['gpu']['index'], seconds=time.perf_counter() - started,
                    shape=list(actions.shape)), actions
    finally:
        policy.close()


def write_progress(output, replicas, requested, started):
    summaries = {}
    for replica in replicas:
        path = Path(replica['directory']) / 'rollouts/summary.json'
        try:
            summaries[str(replica['gpu']['index'])] = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    completed = sum(row['completed_episodes'] for row in summaries.values())
    progress = dict(complete=False, completed_episodes=completed, requested_episodes=requested,
                    evaluation_wall_seconds=time.perf_counter() - started, replicas=summaries)
    temporary = output / 'progress.json.tmp'
    temporary.write_text(json.dumps(progress, indent=2))
    temporary.replace(output / 'progress.json')
    return completed


def validate_policy_config(config):
    """Native OpenPI and SF checkpoints share the simulator/action contract."""
    validate_openpi_config(config)
    if (config.get('backend') not in ('pi0', 'pi05', 'spatial_forcing')
            or config.get('input_contract') not in OPENPI_NATIVE_CONTRACTS):
        raise ValueError('Requires a native pi0/pi05/Spatial Forcing RoboCasa contract')


def main(default_config='pi05.example.json', default_project=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--project', type=Path, default=default_project or ROOT.parent / 'pi05-experiments')
    parser.add_argument('--policy-python', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'evaluation/configs' / default_config)
    parser.add_argument('--workers-per-gpu', type=int, default=8)
    parser.add_argument('--port-base', type=int, default=23000)
    parser.add_argument('--timeout-ms', type=int, default=900000)
    parser.add_argument('--startup-timeout', type=float, default=900)
    parser.add_argument('--output', type=Path, required=True)
    add_benchmark_arguments(parser)
    args = parser.parse_args()
    registry = json.loads((ROOT / 'evaluation/configs/tasks.json').read_text())
    tasks = list(registry) if args.tasks == ['all'] else args.tasks
    if (not args.gpus or len(set(args.gpus)) != len(args.gpus) or
            len(set(tasks)) != len(tasks) or any(task not in registry for task in tasks) or
            len(tasks) < len(args.gpus) or not 1 <= args.workers_per_gpu <= 8):
        parser.error('Need distinct GPUs/tasks, at least one task per GPU, and 1..8 workers per GPU')
    args.seeds = resolve_evaluation_seeds(args)
    specs = episode_specs(tasks, args.episodes, args.seeds, registry)
    if args.replan_steps <= 0:
        parser.error('replan-steps must be positive')
    config = json.loads(args.config.read_text())
    try:
        validate_policy_config(config)
    except ValueError as error:
        parser.error(str(error))
    if args.replan_steps > config['action_horizon']:
        parser.error('Replan interval exceeds model horizon')
    groups = assign_tasks(tasks, registry, len(args.gpus))
    cpu_groups = cpu_partitions(len(args.gpus))
    if args.dry_run:
        print(json.dumps(dict(backend=config['backend'], train_config=config['train_config'],
                              requested_episodes=len(specs), evaluation_seeds=args.seeds,
                              workers_per_gpu=args.workers_per_gpu,
                              tasks_by_gpu=dict(zip(args.gpus, groups))), indent=2))
        return
    args.project = args.project.resolve()
    args.checkpoint = args.checkpoint.resolve()
    policy_python = (args.policy_python or args.project / '.venv/bin/python').absolute()
    required = [policy_python, args.checkpoint / 'params', args.checkpoint / 'assets', args.config]
    if any(not path.exists() for path in required):
        parser.error(f'Missing paths: {[str(path) for path in required if not path.exists()]}')
    if not os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES'):
        parser.error('Source evaluation/egl_env.sh first')
    if not 1024 <= args.port_base <= 65535 - len(args.gpus) + 1:
        parser.error('Invalid service port range')
    revisions = check_runtime()
    inventory = gpu_inventory()
    for rank, gpu in enumerate(args.gpus):
        if gpu not in inventory or inventory[gpu]['free_mib'] < 40000:
            parser.error(f'GPU {gpu} missing or has less than 40 GiB free')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port_base + rank))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'launcher.pid').write_text(str(os.getpid()) + '\n')
    replicas = []
    manager = Processes()
    total_started = time.perf_counter()
    # Graceful cancellation also tears down owned model and worker process groups.
    def terminate(signum, frame):
        (output / 'error.json').write_text(json.dumps(dict(error=f'Received signal {signum}'), indent=2))
        manager.close()
        # Executor context shutdown would otherwise wait for long RPC timeouts.
        os._exit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        for rank, gpu in enumerate(args.gpus):
            directory = output / 'replicas' / f'gpu_{gpu}'
            directory.mkdir(parents=True)
            replica = dict(rank=rank, gpu=inventory[gpu], directory=str(directory),
                           endpoint=f'tcp://127.0.0.1:{args.port_base + rank}',
                           tasks=groups[rank], workers=args.workers_per_gpu,
                           model_cpus=cpu_groups[rank][0], simulator_cpus=cpu_groups[rank][1])
            replicas.append(replica)
            local_config = dict(config, project=str(args.project), checkpoint=str(args.checkpoint))
            (directory / 'policy.json').write_text(json.dumps(local_config, indent=2))
            environment = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               CUDA_VISIBLE_DEVICES=inventory[gpu]['uuid'],
                               XLA_PYTHON_CLIENT_PREALLOCATE='false')
            manager.start(f'model:{gpu}', ['taskset', '-c', ','.join(map(str, replica['model_cpus'])),
                          str(policy_python), '-m', 'evaluation.server',
                          '--config', str(directory / 'policy.json'), '--bind', replica['endpoint']],
                          environment, directory / 'model.log')
        manifest = dict(backend=config['backend'], project=str(args.project), checkpoint=str(args.checkpoint),
                        config=config, revisions=revisions, tasks=tasks,
                        trials_per_task_per_seed=args.episodes, evaluation_seeds=args.seeds,
                        requested_episodes=len(specs), benchmark_protocol=benchmark_defaults(),
                        replan_steps=args.replan_steps, workers_per_gpu=args.workers_per_gpu,
                        max_workers_per_task=args.workers_per_gpu, video=False, replicas=replicas)
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        ready = set()
        deadline = time.monotonic() + args.startup_timeout
        while len(ready) < len(replicas):
            manager.check()
            if time.monotonic() > deadline:
                raise TimeoutError('Model startup timed out')
            for replica in replicas:
                if replica['rank'] in ready:
                    continue
                log = Path(replica['directory']) / 'model.log'
                if 'Policy ready:' in log.read_text():
                    policy = RemotePolicy(replica['endpoint'], 5000)
                    try:
                        if not policy.metadata.get('parallel_safe'):
                            raise RuntimeError('Replica does not support isolated episodes')
                    finally:
                        policy.close()
                    ready.add(replica['rank'])
                    print(f"GPU {replica['gpu']['index']} READY ({len(ready)}/{len(replicas)})", flush=True)
            time.sleep(.5)
        with ThreadPoolExecutor(max_workers=len(replicas)) as executor:
            warmups = list(executor.map(lambda replica: warmup_replica(replica, args.timeout_ms), replicas))
        warmup_info = []
        for info, actions in warmups:
            info['max_action_difference_from_first_replica'] = float(np.max(np.abs(actions - warmups[0][1])))
            warmup_info.append(info)
        (output / 'warmup.json').write_text(json.dumps(warmup_info, indent=2))
        (output / 'gpu_memory_after_warmup.json').write_text(json.dumps(gpu_inventory(), indent=2))
        print('All replicas warmed; starting complete task groups', flush=True)
        evaluation_started = time.perf_counter()
        clients = []
        for replica in replicas:
            environment = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                               MUJOCO_EGL_DEVICE_ID=str(replica['gpu']['index']))
            command = ['taskset', '-c', ','.join(map(str, replica['simulator_cpus'])),
                       sys.executable, '-m', 'evaluation.client', '--endpoint', replica['endpoint'],
                       '--tasks', *replica['tasks'], '--episodes', str(args.episodes),
                       '--seeds', *map(str, args.seeds), '--workers', str(args.workers_per_gpu),
                       '--replan-steps', str(args.replan_steps), '--action-bounds', 'controller',
                       '--timeout-ms', str(args.timeout_ms),
                       '--output', str(Path(replica['directory']) / 'rollouts')]
            clients.append(manager.start(f"client:{replica['gpu']['index']}", command, environment,
                                         Path(replica['directory']) / 'client.log'))
        last_update = 0.
        while any(client.poll() is None for client in clients):
            manager.check()
            if time.monotonic() - last_update >= 15:
                completed = write_progress(output, replicas, len(specs), evaluation_started)
                print(f'Completed {completed}/{len(specs)} episodes', flush=True)
                last_update = time.monotonic()
            time.sleep(1)
        manager.check()
        expected = {(spec['task'], spec['seed']) for spec in specs}
        summary = merge_results(output, replicas, expected, time.perf_counter() - evaluation_started,
                                time.perf_counter() - total_started)
        summary.update(evaluation_seeds=args.seeds, trials_per_task_per_seed=args.episodes,
                       workers_per_gpu=args.workers_per_gpu, max_workers_per_task=args.workers_per_gpu)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        (output / 'progress.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        (output / 'error.json').write_text(json.dumps(dict(error=repr(error)), indent=2))
        raise
    finally:
        manager.close()


if __name__ == '__main__':
    main()
