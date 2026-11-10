"""Asset-backed EGL smoke test without a policy model."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='OpenDrawer')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', args.gpu)
    import numpy as np
    from PIL import Image
    from evaluation.client import check_runtime
    from evaluation.environment import create_env, base_transform, observe
    from evaluation.protocol import CAMERAS

    args.output.mkdir(parents=True, exist_ok=False)
    report = {'task': args.task, 'seed': args.seed, 'passed': False, 'model_loaded': False}
    env = None
    started = time.perf_counter()
    try:
        report['revisions'] = check_runtime()
        print('runtime verified; creating environment', flush=True)
        env = create_env(args.task, args.seed)
        print('environment created; resetting', flush=True)
        env.reset()
        base = base_transform(env)
        before = observe(env, base, args.task, 0, 0)
        report['instruction'] = before['instruction']
        report['cameras'] = {}
        tiles = []
        for name in CAMERAS:
            image = before['images'][name]
            assert image.shape == (256, 256, 3) and image.dtype == np.uint8, name
            assert np.ptp(image) > 0, f'Constant render: {name}'
            Image.fromarray(image).save(args.output / f'{name}.png')
            tiles.append(image)
            report['cameras'][name] = {'shape': list(image.shape), 'dtype': str(image.dtype),
                                       'min': int(image.min()), 'max': int(image.max())}
        Image.fromarray(np.concatenate(tiles, axis=1)).save(args.output / 'cameras.png')
        for key, dimension in [('state8', 8), ('state13', 13)]:
            assert before[key].shape == (dimension,) and np.isfinite(before[key]).all(), key
            report[key] = before[key].tolist()
        assert np.isclose(np.linalg.norm(before['state8'][3:7]), 1.), 'Quaternion is not unit length'
        low, high = env.action_spec
        action = np.zeros_like(low)
        action[0] = .1  # bounded base-frame x control, all non-arm controls remain zero
        assert np.all(action >= low) and np.all(action <= high)
        t0 = float(env.sim.data.time)
        env.step(action)
        after = observe(env, base, args.task, 0, 1)
        assert np.isfinite(env.sim.data.qpos).all() and np.isfinite(after['state8']).all()
        elapsed = float(env.sim.data.time) - t0
        assert np.isclose(elapsed, 1. / env.control_freq, atol=1e-6), elapsed
        report.update(action_dim=len(action), action=action.tolist(), control_freq=env.control_freq,
                      simulated_seconds=elapsed, state8_after=after['state8'].tolist(),
                      task_success=bool(env._check_success()), passed=True)
        print('reset, three-camera render, EE state and one control step passed', flush=True)
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        if env is not None:
            env.close()
        report['wall_seconds'] = time.perf_counter() - started
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
