"""CPU-only Spatial Forcing contract checks; no model/GPU/simulator loading."""
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from evaluation.adapters.openpi import OpenPiPolicy
from evaluation.pi05_multigpu import main, validate_policy_config
from evaluation.protocol import CAMERAS, OPENPI_IMAGE_KEYS, encode_openpi, validate_openpi_config
from evaluation.tests.test_evaluation import FakeEnv, FakePolicy
from evaluation.client import rollout


CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def config():
    return json.loads((CONFIGS / 'spatial_forcing.example.json').read_text())


class NativeSFData:
    repo_id = 'robocasa/robocasa24'
    assets = SimpleNamespace(asset_id=None)


@contextmanager
def runtime(*, pi05=True, align=True, data=None, horizon=50):
    train = SimpleNamespace(data=NativeSFData() if data is None else data,
                            align_enabled=align, align_target_model='vggt',
                            model=SimpleNamespace(pi05=pi05, action_horizon=horizon, action_dim=32))
    stats = {key: SimpleNamespace(mean=np.zeros(dim), std=np.ones(dim),
                                 q01=-np.ones(dim), q99=np.ones(dim))
             for key, dim in [('state', 10), ('actions', 7)]}
    load_stats = Mock(return_value=stats)
    get_config = Mock(return_value=train)
    infer = Mock(side_effect=lambda obs, **kw: {'actions': kw['noise'][:, :7].copy()})
    create = Mock(return_value=SimpleNamespace(infer=infer))
    modules = {
        'openpi': SimpleNamespace(),
        'openpi.policies': SimpleNamespace(policy_config=SimpleNamespace(create_trained_policy=create)),
        'openpi.training': SimpleNamespace(config=SimpleNamespace(
            get_config=get_config, RoboCasaDataConfig=NativeSFData)),
        'openpi.training.checkpoints': SimpleNamespace(load_norm_stats=load_stats),
        # No inference path may require VGGT or training feature-cache modules.
        'vggt': None, 'openpi.training.align_utils': None,
    }
    with patch.dict(sys.modules, modules):
        yield SimpleNamespace(train=train, stats=stats, load_stats=load_stats,
                              get_config=get_config, create=create, infer=infer)


def observation(episode):
    images = {}
    for i, camera in enumerate(CAMERAS):
        images[camera] = np.full((256, 256, 3), [10 + i, 20 + i, 30 + i], np.uint8)
    return dict(**episode, state10=np.arange(10, dtype=np.float32),
                instruction='close the cabinet doors', images=images, step_id=0)


class SpatialForcingContractTests(unittest.TestCase):
    def test_initialization_uses_checkpoint_stats_and_native_public_inputs(self):
        cfg = config(); cfg['checkpoint'] = '/tmp/sf-trained/19999'
        episode = dict(task='CloseDoubleDoor', episode_id=0, seed=7)
        with runtime() as fake:
            adapter = OpenPiPolicy(cfg)
            self.assertTrue(adapter.parallel_safe)
            fake.load_stats.assert_called_once_with(Path('/tmp/sf-trained/19999/assets'), 'robocasa/robocasa24')
            self.assertIs(fake.create.call_args.kwargs['norm_stats'], fake.stats)
            self.assertEqual(fake.create.call_args.kwargs['sample_kwargs'], {'num_steps': 10})
            adapter.reset(episode)
            result = adapter.predict(observation(episode))
            self.assertEqual(result.shape, (50, 7))
            encoded = fake.infer.call_args.args[0]
            self.assertEqual(set(encoded), {'state', 'image', 'prompt'})
            self.assertEqual(tuple(encoded['image']), OPENPI_IMAGE_KEYS)
            for i, image in enumerate(encoded['image'].values()):
                self.assertEqual(image.shape, (224, 224, 3))
                np.testing.assert_array_equal(image[0, 0], [10 + i, 20 + i, 30 + i])
            np.testing.assert_array_equal(encoded['state'], np.arange(10, dtype=np.float32))

    def test_raw_commands_and_gripper_sign_are_unchanged(self):
        episode = dict(task='CloseDoubleDoor', episode_id=0, seed=7)
        commands = np.linspace(-1, 1, 350, dtype=np.float32).reshape(50, 7)
        with runtime() as fake:
            fake.infer.side_effect = None
            fake.infer.return_value = {'actions': commands}
            adapter = OpenPiPolicy(config()); adapter.reset(episode)
            np.testing.assert_array_equal(adapter.predict(observation(episode)), commands)
        policy = FakePolicy(); policy.metadata = {'native_action_tail': config()['native_action_tail']}
        env = FakeEnv()
        rollout(env, policy, 'test', 0, 0, 10, 2)
        np.testing.assert_array_equal(env.actions[0][7:], [0, 0, 0, 0, -1])

    def test_reset_and_interleaving_preserve_episode_sampling(self):
        a = dict(task='CloseDoubleDoor', episode_id=0, seed=17)
        b = dict(task='CloseDoubleDoor', episode_id=1, seed=18)
        with runtime():
            adapter = OpenPiPolicy(config()); adapter.reset(a)
            first = adapter.predict(observation(a)); second = adapter.predict(observation(a))
            adapter.reset(a)
            np.testing.assert_array_equal(adapter.predict(observation(a)), first)
            adapter.reset(b); other = adapter.predict(observation(b))
            np.testing.assert_array_equal(adapter.predict(observation(a)), second)
            self.assertFalse(np.array_equal(first, other))

    def test_wrong_training_config_rejected_before_loading(self):
        for options in ({'pi05': False}, {'align': False}, {'data': object()}, {'horizon': 32}):
            with self.subTest(options=options), runtime(**options) as fake:
                with self.assertRaises(ValueError):
                    OpenPiPolicy(config())
                fake.load_stats.assert_not_called(); fake.create.assert_not_called()

    def test_missing_or_invalid_checkpoint_stats_never_fall_back(self):
        for field in ('mean', 'std', 'q01', 'q99'):
            with self.subTest(field=field), runtime() as fake:
                setattr(fake.stats['actions'], field, np.full(7, np.nan))
                with self.assertRaises(ValueError):
                    OpenPiPolicy(config())
                fake.create.assert_not_called()
        with runtime() as fake:
            fake.load_stats.side_effect = FileNotFoundError('checkpoint stats absent')
            with self.assertRaises(FileNotFoundError):
                OpenPiPolicy(config())
            fake.create.assert_not_called()

    def test_wrong_contract_rejected(self):
        for changes in ({'backend': 'pi05'}, {'state_representation': 'state8'},
                        {'input_contract': 'robocasa_openpi_raw7_v1'}, {'gripper_sign': -1},
                        {'native_action_tail': [0, 0, 0, 0, 0]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_openpi_config(dict(config(), **changes))

    def test_multigpu_dry_run_matches_existing_3600_rollout_plan(self):
        plans = []
        for name in ('spatial_forcing.example.json', 'pi05_baseline.example.json'):
            output = io.StringIO()
            argv = ['evaluation', '--checkpoint', '/not/loaded', '--output', '/not/created', '--dry-run']
            with patch.object(sys, 'argv', argv), redirect_stdout(output), \
                    patch('evaluation.pi05_multigpu.gpu_inventory', side_effect=AssertionError('GPU queried')):
                main(default_config=name)
            plans.append(json.loads(output.getvalue()))
        self.assertEqual(plans[0]['backend'], 'spatial_forcing')
        self.assertEqual(plans[0]['requested_episodes'], 3600)
        self.assertEqual(plans[0]['tasks_by_gpu'], plans[1]['tasks_by_gpu'])
        self.assertEqual(plans[0]['evaluation_seeds'], [0, 1, 2])
        validate_policy_config(config())

    def test_server_script_uses_sf_python_and_forwards_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); project = root / 'project'; project.mkdir()
            step = root / '19'; (step / 'params').mkdir(parents=True); (step / 'assets').mkdir()
            cache = root / 'cache'; (cache / 'big_vision').mkdir(parents=True)
            (cache / 'big_vision/paligemma_tokenizer.model').write_text('test')
            fake_python = root / 'python'
            fake_python.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@"\n'); fake_python.chmod(0o755)
            environment = dict(os.environ, CHECKPOINT=str(step), SF_PROJECT=str(project),
                               SF_PYTHON=str(fake_python), OPENPI_DATA_HOME=str(cache), GPU_ID='0')
            result = subprocess.run(['bash', str(CONFIGS.parent / 'run_spatial_forcing_server.sh'),
                                     '--bind', 'tcp://127.0.0.1:9876'], env=environment,
                                    capture_output=True, text=True, check=True)
            args = result.stdout.splitlines()
            self.assertEqual(args[:2], ['-m', 'evaluation.server'])
            self.assertIn(str(step), args)
            self.assertIn(str(project), args)
            self.assertEqual(args[-2:], ['--bind', 'tcp://127.0.0.1:9876'])


if __name__ == '__main__':
    unittest.main()
