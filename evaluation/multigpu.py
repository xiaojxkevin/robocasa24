"""Managed H20 X-WAM replicas with independent endpoints and global episode shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import numpy as np
from evaluation.client import ROOT, episode_specs, episode_plan, success_metrics
from evaluation.protocol import benchmark_defaults, add_benchmark_arguments, resolve_evaluation_seeds
from evaluation.transport import RemotePolicy


class Processes:
    def __init__(self):
        self.items = []

    def start(self, name, command, environment, log):
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        self.items.append((name, process, log))
        return process

    def check(self):
        for name, process, log in self.items:
            code = process.poll()
            if code is not None and (not name.startswith('client:') or code != 0):
                raise RuntimeError(f'{name} exited {code}; see {log}')

    def close(self):
        for _, process, _ in reversed(self.items):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _, process, _ in self.items:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def gpu_inventory():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    inventory = {}
    for line in result.splitlines():
        index, uuid, name, total, free = [value.strip() for value in line.split(',')]
        inventory[int(index)] = dict(index=int(index), uuid=uuid, name=name, total_mib=int(total), free_mib=int(free))
    return inventory


def warmup_observation(dataset):
    import imageio.v2 as imageio
    from scipy.spatial.transform import Rotation
    from evaluation.protocol import CAMERAS
    path = next(iter(sorted((dataset / 'data').rglob('episode_*.json'))))
    episode = json.loads(path.read_text())
    images = {}
    for camera in CAMERAS:
        reader = imageio.get_reader(dataset / episode['observations'][camera]['rgb_path'])
        try:
            images[camera] = reader.get_data(0)
        finally:
            reader.close()
    proprios = episode['proprios']
    quat = Rotation.from_matrix(np.asarray(proprios['left_ee_rotm'][0]).reshape(3, 3)).as_quat()[[3, 0, 1, 2]]
    nonzero = quat[np.flatnonzero(quat)]
    if len(nonzero) and nonzero[0] < 0:
        quat = -quat
    state = np.concatenate([proprios['left_ee_pos'][0], quat, proprios['left_gripper_pos'][0]]).astype(np.float32)
    return dict(images=images, state8=state, instruction=episode['instructions'][0], episode_id=0, step_id=0)


def warmup_replica(replica, observation, timeout_ms):
    policy = RemotePolicy(replica['endpoint'], timeout_ms)
    try:
        started = time.perf_counter()
        actions = policy.predict(observation)
        return dict(gpu=replica['gpu']['index'], seconds=time.perf_counter() - started,
                    shape=list(actions.shape)), actions
    finally:
        policy.close()


def merge_results(output, replicas, expected, evaluation_seconds, total_seconds):
    by_task = {}
    seen = set()
    for replica in replicas:
        for path in (Path(replica['directory']) / 'rollouts').glob('*/episodes.json'):
            task = path.parent.name
            for record in json.loads(path.read_text()):
                key = (task, record['seed'])
                if key in seen or key not in expected:
                    raise ValueError(f'Duplicate or unexpected episode: {key}')
                seen.add(key)
                record['replica_gpu'] = replica['gpu']['index']
                by_task.setdefault(task, []).append(record)
    if seen != expected:
        raise ValueError(f'Missing {len(expected - seen)} episodes')
    for task, records in by_task.items():
        directory = output / task
        directory.mkdir()
        records.sort(key=lambda r: r['seed'])
        (directory / 'episodes.json').write_text(json.dumps(records, indent=2))
    records = [r for rows in by_task.values() for r in rows]
    summary = dict(complete=True, **success_metrics(by_task),
                   completed_tasks=len(by_task), requested_tasks=len(by_task), completed_episodes=len(seen),
                   requested_episodes=len(expected), success_count=sum(r['success'] for r in records),
                   steps=sum(r['steps'] for r in records), replicas=len(replicas),
                   run_wall_seconds=evaluation_seconds, total_wall_seconds=total_seconds,
                   startup_and_warmup_seconds=total_seconds - evaluation_seconds,
                   episodes_per_minute=60 * len(seen) / evaluation_seconds)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--project', type=Path, default=ROOT.parent / 'X-WAM')
    parser.add_argument('--policy-python', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--wan-checkpoint', type=Path)
    parser.add_argument('--dataset', type=Path)
    add_benchmark_arguments(parser)
    parser.add_argument('--workers-per-gpu', type=int, default=8)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--video-stride', type=int, default=4)
    parser.add_argument('--port-base', type=int, default=22000)
    parser.add_argument('--timeout-ms', type=int, default=900000)
    parser.add_argument('--startup-timeout', type=float, default=900)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.project = args.project.resolve()
    policy_python = (args.policy_python or args.project / '.venv/bin/python').absolute()
    checkpoint = (args.checkpoint or args.project / 'checkpoints/robocasa_sft').resolve()
    wan = (args.wan_checkpoint or args.project / 'checkpoints/Wan2.2-TI2V-5B').resolve()
    dataset = (args.dataset or args.project / 'datasets/RoboCasa').resolve()
    registry = json.loads((ROOT / 'evaluation/configs/tasks.json').read_text())
    tasks = list(registry) if args.tasks == ['all'] else args.tasks
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or len(set(tasks)) != len(tasks):
        parser.error('GPU indices and task names must be distinct')
    if any(task not in registry for task in tasks) or min(args.episodes, args.workers_per_gpu, args.replan_steps, args.video_stride) <= 0:
        parser.error('Unknown task or nonpositive count')
    try:
        args.seeds = resolve_evaluation_seeds(args)
        specs = episode_specs(tasks, args.episodes, args.seeds, registry)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(dict(**episode_plan(tasks, args.episodes, args.seeds, specs),
                              gpus=args.gpus, workers_per_gpu=args.workers_per_gpu), indent=2))
        return
    if len(specs) < len(args.gpus):
        parser.error('Need at least one episode per selected GPU')
    if not os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES'):
        parser.error('Source evaluation/egl_env.sh before launching')
    required = [policy_python, checkpoint / 'checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt',
                wan / 'Wan2.2_VAE.pth', dataset / 'metadata.json']
    if not all(p.exists() for p in required):
        parser.error(f'Missing input paths: {[str(p) for p in required if not p.exists()]}')
    if not 1024 <= args.port_base or args.port_base + 3 * len(args.gpus) > 65535:
        parser.error('Invalid port range')
    inventory = gpu_inventory()
    for gpu in args.gpus:
        if gpu not in inventory or 'H20' not in inventory[gpu]['name']:
            parser.error(f'This verified configuration requires an H20: {gpu}')
        if inventory[gpu]['free_mib'] < 60000:
            parser.error(f'GPU {gpu} has insufficient free memory for this configuration')
    # Fail before loading weights if any of the planned service ports is occupied.
    ports = []
    try:
        for port in range(args.port_base, args.port_base + 3 * len(args.gpus)):
            sock = socket.socket()
            ports.append(sock)
            sock.bind(('127.0.0.1', port))
    finally:
        for sock in ports:
            sock.close()
    observation = warmup_observation(dataset)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    replicas = []
    manager = Processes()
    total_started = time.perf_counter()
    try:
        for rank, gpu_index in enumerate(args.gpus):
            directory = output / 'replicas' / f'gpu_{gpu_index}'
            directory.mkdir(parents=True)
            frontend, backend, bridge = [args.port_base + 3 * rank + offset for offset in range(3)]
            endpoint = f'tcp://127.0.0.1:{bridge}'
            environment = os.environ.copy()
            environment.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                               OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='2')
            model_environment = dict(environment, CUDA_VISIBLE_DEVICES=inventory[gpu_index]['uuid'])
            replica = dict(rank=rank, gpu=inventory[gpu_index], directory=str(directory), endpoint=endpoint,
                           frontend=frontend, backend=backend, workers=args.workers_per_gpu)
            replicas.append(replica)
            config = dict(backend='xwam', broker_endpoint=f'tcp://127.0.0.1:{frontend}',
                          action_semantics='robosuite_raw7', timeout_ms=args.timeout_ms, env_rank=0,
                          checkpoint=str(checkpoint), cfg=0.)
            (directory / 'bridge.json').write_text(json.dumps(config, indent=2))
            manager.start(f'broker:{gpu_index}', [str(policy_python), str(args.project / 'evaluation/policy_broker.py'),
                          '--frontend-port', str(frontend), '--backend-port', str(backend), '--host', '127.0.0.1'],
                          environment, directory / 'broker.log')
            manager.start(f'model:{gpu_index}', [str(policy_python), '-m', 'evaluation.xwam_server',
                          '--project', str(args.project), '--exp-path', str(checkpoint), '--wan-checkpoint-dir', str(wan),
                          '--broker-port', str(backend), '--denoise-steps', '50', '--action-denoise-steps', '10'],
                          model_environment, directory / 'model.log')
            manager.start(f'bridge:{gpu_index}', [sys.executable, '-m', 'evaluation.server', '--config',
                          str(directory / 'bridge.json'), '--bind', endpoint], environment, directory / 'bridge.log')
        manifest = dict(checkpoint=str(checkpoint), project=str(args.project), wan_checkpoint=str(wan),
                        tasks=tasks, trials_per_task_per_seed=args.episodes, evaluation_seeds=args.seeds,
                        benchmark_protocol=benchmark_defaults(), replan_steps=args.replan_steps,
                        workers_per_gpu=args.workers_per_gpu, video=args.video, video_stride=args.video_stride,
                        replicas=replicas, cpu_checkpoint_staging=True)
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        deadline = time.monotonic() + args.startup_timeout
        ready = set()
        while len(ready) != len(replicas):
            manager.check()
            if time.monotonic() > deadline:
                raise TimeoutError('Model readiness timeout; inspect replica logs')
            for replica in replicas:
                if replica['rank'] in ready:
                    continue
                log = Path(replica['directory']) / 'broker.log'
                if log.exists() and ' READY ' in log.read_text():
                    policy = RemotePolicy(replica['endpoint'], 2000)
                    policy.close()
                    ready.add(replica['rank'])
                    print(f"GPU {replica['gpu']['index']} model READY ({len(ready)}/{len(replicas)})", flush=True)
            time.sleep(.5)
        with ThreadPoolExecutor(max_workers=len(replicas)) as executor:
            warmups = list(executor.map(lambda r: warmup_replica(r, observation, args.timeout_ms), replicas))
        reference = warmups[0][1]
        metadata = []
        for info, actions in warmups:
            info['max_action_difference_from_first_replica'] = float(np.max(np.abs(actions - reference)))
            metadata.append(info)
        (output / 'warmup.json').write_text(json.dumps(metadata, indent=2))
        (output / 'gpu_memory_after_warmup.json').write_text(json.dumps(gpu_inventory(), indent=2))
        print('All replicas warmed; starting global episode shards', flush=True)
        evaluation_started = time.perf_counter()
        clients = []
        for replica in replicas:
            environment = os.environ.copy()
            environment.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                               MUJOCO_EGL_DEVICE_ID=str(replica['gpu']['index']))
            command = [sys.executable, '-m', 'evaluation.client', '--endpoint', replica['endpoint'], '--tasks', *tasks,
                       '--episodes', str(args.episodes), '--seeds', *map(str, args.seeds), '--workers', str(args.workers_per_gpu),
                       '--shard-count', str(len(replicas)), '--shard-index', str(replica['rank']),
                       '--replan-steps', str(args.replan_steps), '--video-stride', str(args.video_stride),
                       '--timeout-ms', str(args.timeout_ms), '--action-bounds', 'controller',
                       '--output', str(Path(replica['directory']) / 'rollouts')]
            if args.video:
                command.append('--video')
            clients.append(manager.start(f"client:{replica['gpu']['index']}", command, environment,
                                         Path(replica['directory']) / 'client.log'))
        while any(p.poll() is None for p in clients):
            manager.check()
            time.sleep(.5)
        manager.check()
        elapsed = time.perf_counter() - evaluation_started
        expected = {(s['task'], s['seed']) for s in episode_specs(tasks, args.episodes, args.seeds, registry)}
        summary = merge_results(output, replicas, expected, elapsed, time.perf_counter() - total_started)
        summary.update(evaluation_seeds=args.seeds, trials_per_task_per_seed=args.episodes)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        (output / 'error.json').write_text(json.dumps(dict(error=repr(error)), indent=2))
        raise
    finally:
        manager.close()


if __name__ == '__main__':
    main()
