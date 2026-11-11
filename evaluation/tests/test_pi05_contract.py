"""Deferred interface tests; no model/checkpoint required.

Not executed while training is active. These checks cover the new named
contract without changing the legacy policy format.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from evaluation.protocol import (CAMERAS, EEF_AXES, OPENPI_IMAGE_KEYS,
                                 encode_openpi, native_action_template,
                                 validate_openpi_config)
from evaluation.environment import observe, base_transform
from evaluation.adapters.openpi import OpenPiPolicy
from evaluation.tests.test_evaluation import FakeEnv, FakePolicy
from evaluation.client import rollout


def config():
    path = Path(__file__).resolve().parents[1] / 'configs/pi05.example.json'
    return json.loads(path.read_text())


class Pi05ContractTests(unittest.TestCase):
    def test_state10_rotation_columns_and_native_gripper(self):
        env = FakeEnv()
        obs = observe(env, base_transform(env), 'task', 0, 0)
        expected = np.concatenate([[.1, .2, .3], EEF_AXES[:, 0], EEF_AXES[:, 1], [.5]])
        np.testing.assert_allclose(obs['state10'], expected)
        self.assertEqual(obs['state10'].dtype, np.float32)

    def test_named_input_is_public_policy_format_without_auxiliary_lookup(self):
        obs = dict(state10=np.arange(10, dtype=np.float32), instruction='close drawer',
                   episode_id=123, step_id=456,
                   images={c: np.full((256, 256, 3), i, np.uint8) for i, c in enumerate(CAMERAS)})
        encoded = encode_openpi(obs, config())
        self.assertEqual(set(encoded), {'state', 'image', 'prompt'})
        self.assertEqual(tuple(encoded['image']), OPENPI_IMAGE_KEYS)
        for i, image in enumerate(encoded['image'].values()):
            self.assertEqual(image.shape, (224, 224, 3))
            self.assertEqual(image.dtype, np.uint8)
            np.testing.assert_array_equal(image, np.full((224, 224, 3), i, np.uint8))
        np.testing.assert_array_equal(encoded['state'], obs['state10'])

    def test_mismatched_native_contract_rejected(self):
        for name, value in [('state_representation', 'state8'), ('image_layout', 'chw'),
                            ('image_size', 256), ('gripper_sign', -1),
                            ('native_action_tail', [0, 0, 0, 0, 0])]:
            bad = config(); bad[name] = value
            with self.assertRaises(ValueError):
                validate_openpi_config(bad)

    def test_wrong_state_shape_rejected(self):
        with self.assertRaises(ValueError):
            encode_openpi(dict(state10=np.zeros(13)), config())

    def test_legacy_images_contract_unchanged(self):
        obs = dict(state8=np.zeros(8), instruction='test',
                   images={c: np.zeros((256, 256, 3), np.uint8) for c in CAMERAS})
        encoded = encode_openpi(obs, dict(state_representation='state8',
                                        image_keys=['a', 'b', 'c'], image_layout='chw'))
        self.assertIn('images', encoded)
        self.assertEqual(encoded['images']['a'].shape, (3, 256, 256))

    def test_native_tail_and_legacy_padding(self):
        low, high = -np.ones(12), np.ones(12)
        expected = np.zeros(12); expected[-1] = -1
        np.testing.assert_array_equal(native_action_template(low, high, [0, 0, 0, 0, -1]), expected)
        np.testing.assert_array_equal(native_action_template(low, high), np.zeros(12))
        for tail in ([0]*4, [0, 0, 0, 0, -2]):
            with self.assertRaises(ValueError):
                native_action_template(low, high, tail)

    def test_rollout_uses_tail_from_server_metadata(self):
        policy = FakePolicy()
        policy.metadata = {'native_action_tail': [0, 0, 0, 0, -1]}
        env = FakeEnv()
        result = rollout(env, policy, 'task', 0, 0, 10, 2)
        np.testing.assert_array_equal(env.actions[0][7:], [0, 0, 0, 0, -1])
        np.testing.assert_array_equal(env.actions[0][:7], np.full(7, .25))
        self.assertEqual(result['native_action_tail'], [0, 0, 0, 0, -1])

    def test_predict_preserves_native_commands_and_episode_noise(self):
        adapter = OpenPiPolicy.__new__(OpenPiPolicy)
        adapter.config = config(); adapter.native_robocasa = True
        adapter.horizon = 50; adapter.action_dim = 32
        adapter.noise_rng = np.random.default_rng(0)
        commands = np.linspace(-1, 1, 350, dtype=np.float32).reshape(50, 7)
        noise = []
        def infer(obs, **kwargs):
            noise.append(kwargs['noise'].copy())
            return {'actions': commands.copy()}
        adapter.policy = SimpleNamespace(infer=infer)
        with patch('evaluation.adapters.openpi.encode_openpi', return_value={}):
            adapter.reset({'seed': 7}); result = adapter.predict({})
            adapter.reset({'seed': 7}); adapter.predict({})
        np.testing.assert_array_equal(result, commands)
        np.testing.assert_array_equal(noise[0], noise[1])
        self.assertEqual(noise[0].shape, (50, 32))


if __name__ == '__main__':
    unittest.main()
