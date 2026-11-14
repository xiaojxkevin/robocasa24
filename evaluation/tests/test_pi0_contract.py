"""Native pi0/pi05 initialization and compatibility, without allocating a model."""
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from evaluation.adapters.openpi import OpenPiPolicy
from evaluation.pi05_multigpu import main, validate_policy_config
from evaluation.protocol import CAMERAS, encode_openpi


CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
FILES = ('pi05.example.json', 'pi0.example.json', 'pi0_baseline.example.json',
         'pi05_baseline.example.json')


def read_config(name):
    return json.loads((CONFIGS / name).read_text())


class NativeData:
    repo_id = 'robocasa/robocasa24'
    assets = SimpleNamespace(asset_id=None)


@contextmanager
def runtime(config, *, pi05=None, rtc=False, horizon=50, state_dim=10):
    """Fake only OpenPI imports; execute the real adapter initialization."""
    train = SimpleNamespace(
        data=NativeData(), model=SimpleNamespace(
            pi05=config['backend'] == 'pi05' if pi05 is None else pi05,
            rtc=rtc, action_horizon=horizon, action_dim=32))
    stats = {key: SimpleNamespace(mean=np.zeros(dim))
             for key, dim in [('state', state_dim), ('actions', 7)]}
    load_stats = Mock(return_value=stats)
    get_config = Mock(return_value=train)
    infer = Mock(return_value={'actions': np.zeros((50, 7), np.float32)})
    create = Mock(return_value=SimpleNamespace(infer=infer))
    training = SimpleNamespace(get_config=get_config, RoboCasaD4rtDataConfig=NativeData)
    modules = {
        'openpi': SimpleNamespace(),
        'openpi.policies': SimpleNamespace(policy_config=SimpleNamespace(create_trained_policy=create)),
        'openpi.training': SimpleNamespace(config=training),
        'openpi.training.checkpoints': SimpleNamespace(load_norm_stats=load_stats),
    }
    with patch.dict(sys.modules, modules):
        yield SimpleNamespace(create=create, load_stats=load_stats, stats=stats, get_config=get_config,
                              infer=infer, train=train)


class Pi0ContractTests(unittest.TestCase):
    def test_all_four_configs_initialize_and_preserve_native_actions(self):
        episode = dict(task='CloseDoubleDoor', episode_id=0, seed=10)
        obs = dict(**episode, state10=np.arange(10, dtype=np.float32), instruction='close doors',
                   images={cam: np.zeros((256, 256, 3), np.uint8) for cam in CAMERAS})
        for name in FILES:
            with self.subTest(config=name):
                config = read_config(name)
                config['checkpoint'] = '/tmp/robocasa-trained-checkpoint/19999'
                validate_policy_config(config)
                with runtime(config) as mocked:
                    adapter = OpenPiPolicy(config)
                    self.assertTrue(adapter.native_robocasa)
                    self.assertTrue(adapter.parallel_safe)
                    mocked.get_config.assert_called_once_with(config['train_config'])
                    mocked.load_stats.assert_called_once_with(
                        Path(config['checkpoint']) / 'assets', 'robocasa/robocasa24')
                    self.assertIs(mocked.create.call_args.kwargs['norm_stats'], mocked.stats)
                    self.assertEqual(mocked.create.call_args.kwargs['sample_kwargs'], {'num_steps': 10})
                    commands = np.linspace(-1, 1, 350, dtype=np.float32).reshape(50, 7)
                    mocked.infer.return_value = {'actions': commands}
                    adapter.reset(episode)
                    np.testing.assert_array_equal(adapter.predict(obs), commands)
                    noise = mocked.infer.call_args.kwargs['noise'].copy()
                    adapter.reset(episode)
                    adapter.predict(obs)
                    np.testing.assert_array_equal(mocked.infer.call_args.kwargs['noise'], noise)
                    self.assertEqual(noise.shape, (50, 32))
                    encoded = mocked.infer.call_args.args[0]
                    self.assertEqual(set(encoded), {'state', 'image', 'prompt'})
                    np.testing.assert_array_equal(encoded['state'], obs['state10'])

    def test_backend_model_mismatches_and_rtc_are_rejected_before_loading(self):
        for name in FILES:
            for options in ({'pi05': read_config(name)['backend'] != 'pi05'}, {'rtc': True}):
                with self.subTest(config=name, options=options):
                    config = read_config(name)
                    with runtime(config, **options) as mocked:
                        with self.assertRaises(ValueError):
                            OpenPiPolicy(config)
                        mocked.load_stats.assert_not_called()
                        mocked.create.assert_not_called()

    def test_bad_checkpoint_stats_and_missing_stats_do_not_fall_back(self):
        for name in FILES:
            with self.subTest(config=name):
                config = read_config(name)
                with runtime(config, state_dim=8) as mocked:
                    with self.assertRaisesRegex(ValueError, 'state dimension 10'):
                        OpenPiPolicy(config)
                    mocked.create.assert_not_called()
                with runtime(config) as mocked:
                    mocked.load_stats.side_effect = FileNotFoundError('missing checkpoint stats')
                    with self.assertRaises(FileNotFoundError):
                        OpenPiPolicy(config)
                    mocked.create.assert_not_called()

    def test_contract_aliases_encode_identical_public_inputs(self):
        obs = dict(state10=np.arange(10), instruction='test',
                   images={cam: np.zeros((224, 224, 3), np.uint8) for cam in CAMERAS})
        original = encode_openpi(obs, read_config(FILES[0]))
        for name in FILES[1:]:
            with self.subTest(config=name):
                encoded = encode_openpi(obs, read_config(name))
                self.assertEqual(set(encoded), set(original))
                np.testing.assert_array_equal(encoded['state'], original['state'])
                for key in original['image']:
                    np.testing.assert_array_equal(encoded['image'][key], original['image'][key])

    def test_all_dry_runs_share_the_same_3600_rollout_plan(self):
        reference = None
        for name in FILES:
            with self.subTest(config=name):
                output = io.StringIO()
                argv = ['evaluation', '--checkpoint', '/nonexistent/step',
                        '--output', '/nonexistent/results', '--dry-run']
                with patch.object(sys, 'argv', argv), redirect_stdout(output):
                    main(default_config=name)
                plan = json.loads(output.getvalue())
                self.assertEqual(plan['backend'], read_config(name)['backend'])
                self.assertEqual(plan['train_config'], read_config(name)['train_config'])
                self.assertEqual(plan['requested_episodes'], 3600)
                self.assertEqual(plan['evaluation_seeds'], [0, 1, 2])
                if reference is None:
                    reference = plan['tasks_by_gpu']
                self.assertEqual(plan['tasks_by_gpu'], reference)

    def test_multigpu_rejects_legacy_or_other_backends(self):
        for changes in ({'input_contract': 'legacy'}, {'backend': 'xwam'},
                        {'state_representation': 'state8'}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    validate_policy_config(dict(read_config('pi0.example.json'), **changes))


if __name__ == '__main__':
    unittest.main()
