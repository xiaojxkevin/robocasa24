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
OPENPI_ROBOCASA_CONTRACT = 'robocasa_pi05_raw7_v1'
OPENPI_IMAGE_KEYS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')
NATIVE_ACTION_TAIL = (0., 0., 0., 0., -1.)


def validate_actions(actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError(f'Expected nonempty [T,7] native actions; got {actions.shape}')
    if not np.isfinite(actions).all():
        raise ValueError('Nonfinite action')
    return actions


def validate_openpi_config(config):
    """Validate the named training contract without importing a model runtime."""
    contract = config.get('input_contract', 'legacy')
    if contract not in ('legacy', OPENPI_ROBOCASA_CONTRACT):
        raise ValueError(f'Unknown OpenPI input_contract: {contract}')
    keys = config['image_keys']
    if len(keys) != 3 or len(set(keys)) != 3:
        raise ValueError('image_keys must contain three distinct keys in canonical camera order')
    if config.get('image_layout', 'hwc') not in ('hwc', 'chw'):
        raise ValueError('image_layout must be hwc or chw')
    if contract == OPENPI_ROBOCASA_CONTRACT:
        expected = dict(state_representation='state10', image_layout='hwc', image_size=224,
                        action_semantics='robosuite_raw7', gripper_sign=1)
        for name, value in expected.items():
            if config.get(name) != value:
                raise ValueError(f'{contract} requires {name}={value!r}')
        if tuple(keys) != OPENPI_IMAGE_KEYS:
            raise ValueError(f'{contract} requires image_keys={OPENPI_IMAGE_KEYS}')
        if config.get('action_indices') != list(range(7)):
            raise ValueError('Native raw_action dimensions must remain in order 0..6')
        if tuple(config.get('native_action_tail', ())) != NATIVE_ACTION_TAIL:
            raise ValueError(f'{contract} requires native_action_tail={NATIVE_ACTION_TAIL}')


def encode_openpi(obs, config):
    """Return public policy inputs, not LeRobot column names/repack inputs.

    create_trained_policy does not apply the training repack transform. The
    RoboCasa model consumes singular `image` and a 10D state directly. Dataset
    episode/frame IDs are deliberately absent: online inference must not look
    up training depth, tracks or per-episode auxiliary geometry.
    """
    validate_openpi_config(config)
    new_contract = config.get('input_contract') == OPENPI_ROBOCASA_CONTRACT
    state = np.asarray(obs[config['state_representation']], dtype=np.float32).copy()
    if new_contract and (state.shape != (10,) or not np.isfinite(state).all()):
        raise ValueError('Expected finite state10: xyz, rotation columns 0 and 1, gripper')
    images = {}
    for camera, key in zip(CAMERAS, config['image_keys']):
        rgb = np.asarray(obs['images'][camera])
        if new_contract:
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f'Expected HWC uint8 RGB for {camera}, got {rgb.shape}/{rgb.dtype}')
            if rgb.shape[:2] != (224, 224):
                import cv2
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            images[key] = np.ascontiguousarray(rgb)
        else:
            images[key] = rgb.transpose(2, 0, 1) if config.get('image_layout', 'hwc') == 'chw' else rgb
    return {'state': state, 'image' if new_contract else 'images': images,
            'prompt': obs['instruction']}


def native_action_template(low, high, tail=None):
    """Fixed non-arm controls; legacy policies retain zero padding by default."""
    low, high = np.asarray(low), np.asarray(high)
    if low.ndim != 1 or low.shape != high.shape or len(low) < 7:
        raise ValueError('Expected flat controller action limits with at least seven dimensions')
    values = np.zeros(len(low) - 7) if tail is None else np.asarray(tail, dtype=np.float32)
    if values.shape != (len(low) - 7,) or not np.isfinite(values).all():
        raise ValueError(f'Expected finite native_action_tail with {len(low) - 7} components')
    if np.any(values < low[7:]) or np.any(values > high[7:]):
        raise ValueError('Fixed action tail exceeds controller limits')
    result = np.zeros_like(low)
    result[7:] = values
    return result


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
