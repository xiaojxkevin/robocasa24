"""RoboCasa24 v1: RGB uint8, X-WAM base-frame EE state, native raw actions."""
import copy
import json
from pathlib import Path
import numpy as np

PROTOCOL = 'robocasa24.raw.v1'
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
ROBOCASA_COMMIT = '756598a5be52e052339bb2d957426e39015c2afb'
ROBOSUITE_COMMIT = '232ce7d4a6ed89c949a9aba024a05c8c32fdd08b'
EEF_AXES = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def validate_actions(actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError(f'Expected nonempty [T,7] native actions; got {actions.shape}')
    if not np.isfinite(actions).all():
        raise ValueError('Nonfinite action')
    return actions


def encode_openpi(obs, config):
    """Keys/layout are explicit because trained OpenPI data transforms differ."""
    state = obs[config['state_representation']].copy()
    keys = config['image_keys']
    if len(keys) != 3 or len(set(keys)) != 3:
        raise ValueError('image_keys must contain three distinct keys in canonical camera order')
    if config.get('image_layout', 'hwc') not in ('hwc', 'chw'):
        raise ValueError('image_layout must be hwc or chw')
    images = {}
    for camera, key in zip(CAMERAS, keys):
        rgb = obs['images'][camera]
        images[key] = rgb.transpose(2, 0, 1) if config.get('image_layout', 'hwc') == 'chw' else rgb
    return {'state': state, 'images': images, 'prompt': obs['instruction']}


_BENCHMARK = json.loads((Path(__file__).resolve().parent / 'configs/benchmark.json').read_text())
EVALUATION_SEED_BLOCK = _BENCHMARK['evaluation_seed_block']
TASK_SEED_STRIDE = _BENCHMARK['task_seed_stride']


def benchmark_defaults():
    return copy.deepcopy(_BENCHMARK)


def add_benchmark_arguments(parser):
    defaults = benchmark_defaults()
    parser.add_argument('--tasks', nargs='+', default=defaults['tasks'])
    parser.add_argument('--episodes', type=int, default=defaults['trials_per_task_per_seed'],
                        help='Trials PER TASK PER EVALUATION SEED globally, not per GPU')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--seeds', nargs='+', type=int, help='Evaluation seed list; default from benchmark.json')
    group.add_argument('--seed', type=int, help='Single-evaluation-seed shorthand for small runs')
    parser.add_argument('--replan-steps', type=int, default=defaults['replan_steps'])
    parser.add_argument('--dry-run', action='store_true', help='Print the episode plan without starting simulation/models')


def resolve_evaluation_seeds(args):
    seeds = args.seeds if args.seeds is not None else ([args.seed] if args.seed is not None else _BENCHMARK['evaluation_seeds'])
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('Evaluation seeds must be nonempty, distinct and nonnegative')
    return list(seeds)
