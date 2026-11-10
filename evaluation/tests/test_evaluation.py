"""CPU tests for shared execution semantics; no simulator assets/checkpoints."""
import unittest
from types import SimpleNamespace as NS
import numpy as np
from evaluation.protocol import CAMERAS, encode_openpi, validate_actions
from evaluation.environment import observe, base_transform, render_backend
from evaluation.client import rollout, episode_seed, episode_specs
from evaluation.adapters.xwam import XWamPolicy


class FakeEnv:
    def __init__(self, succeed_at=3):
        arm = NS(origin_ori=np.eye(3), origin_pos=np.zeros(3),
                 ref_ori_mat=np.eye(3), ref_pos=np.array([.1, .2, .3]))
        grip = NS(joint_pos=np.array([.02]), actuator_max=np.array([.04]))
        controller = NS(arms=['right'], grippers={'grip': None}, part_controllers={'right': arm, 'grip': grip})
        self.robots = [NS(composite_controller=controller)]
        self.control_freq = 20
        self.action_spec = (-np.ones(12), np.ones(12))
        self.sim = NS(render=lambda **kw: np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
                      get_state=lambda: np.zeros(5))
        self.actions = []
        self.succeed_at = succeed_at

    def reset(self):
        self.actions.clear()

    def step(self, action):
        self.actions.append(action.copy())

    def get_ep_meta(self):
        return {'lang': 'test task'}

    def _check_success(self):
        return len(self.actions) >= self.succeed_at


class FakePolicy:
    def __init__(self, chunk=4):
        self.chunk = chunk
        self.resets = []
        self.steps = []

    def reset(self, episode):
        self.resets.append(episode)

    def predict(self, obs):
        self.steps.append(obs['step_id'])
        return np.full((self.chunk, 7), .25, dtype=np.float32)


class EvaluationTests(unittest.TestCase):
    def test_invalid_actions(self):
        for actions in [[], np.zeros((2, 14)), np.zeros(7), np.full((2, 7), np.nan)]:
            with self.assertRaises(ValueError):
                validate_actions(actions)

    def test_observation_convention(self):
        env = FakeEnv()
        obs = observe(env, base_transform(env), 'test', 7, 0)
        np.testing.assert_allclose(obs['state8'], [.1, .2, .3, 2**-.5, 0, 0, 2**-.5, .5])
        np.testing.assert_array_equal(obs['images'][CAMERAS[0]], env.sim.render()[::-1])
        self.assertEqual(obs['state13'].shape, (13,))

    def test_openpi_mapping(self):
        obs = observe(FakeEnv(), np.eye(4), 'test', 0, 0)
        encoded = encode_openpi(obs, {'state_representation': 'state8', 'image_keys': ['a', 'b', 'c'], 'image_layout': 'chw'})
        self.assertEqual(encoded['images']['a'].shape, (3, 2, 2))
        self.assertEqual(encoded['prompt'], 'test task')

    def test_success_and_base_padding(self):
        env, policy = FakeEnv(), FakePolicy()
        record = rollout(env, policy, 'test', 9, 9, 10, 2)
        self.assertTrue(record['success'])
        self.assertEqual(record['steps'], 3)
        self.assertEqual(policy.steps, [0, 2])
        self.assertEqual(len(policy.resets), 1)
        np.testing.assert_array_equal(env.actions[0][7:], np.zeros(5))

    def test_horizon_is_not_overshot(self):
        env = FakeEnv(succeed_at=100)
        record = rollout(env, FakePolicy(), 'test', 0, 0, 3, 2)
        self.assertFalse(record['success'])
        self.assertEqual(len(env.actions), 3)

    def test_short_chunk_rejected(self):
        with self.assertRaises(ValueError):
            rollout(FakeEnv(), FakePolicy(chunk=1), 'test', 0, 0, 10, 2)

    def test_out_of_bounds_rejected(self):
        policy = FakePolicy()
        policy.predict = lambda obs: np.full((4, 7), 2.)
        with self.assertRaises(ValueError):
            rollout(FakeEnv(), policy, 'test', 0, 0, 10, 2)

    def test_controller_bounds_mode_preserves_raw_controls(self):
        policy = FakePolicy()
        policy.predict = lambda obs: np.full((4, 7), 1.01)
        env = FakeEnv()
        record = rollout(env, policy, 'test', 0, 0, 10, 2, action_bounds='controller')
        self.assertEqual(record['out_of_bounds_components'], 21)
        self.assertAlmostEqual(record['max_bound_violation'], .01, places=6)
        np.testing.assert_allclose(env.actions[0][:7], 1.01)

    def test_video_stride_does_not_change_actions(self):
        from unittest.mock import patch
        writer = NS(append_data=lambda frame: None, close=lambda: None)
        env = FakeEnv(succeed_at=9)
        with patch('imageio.v2.get_writer', return_value=writer) as factory:
            record = rollout(env, FakePolicy(), 'test', 0, 0, 10, 2, 'test.mp4', video_stride=4)
        self.assertEqual(record['steps'], 9)
        self.assertEqual(record['video_frames'], 3)
        self.assertEqual(record['video_fps'], 5)
        factory.assert_called_once_with('test.mp4', fps=5)

    def test_seed_independent_of_job_order(self):
        registry = {'a': 500, 'b': 500}
        self.assertEqual(episode_seed('b', 2, 0, registry), 100002)
        jobs = [('a', 0), ('b', 2), ('b', 1)]
        forward = {(t, i): episode_seed(t, i, 7, registry) for t, i in jobs}
        backward = {(t, i): episode_seed(t, i, 7, registry) for t, i in reversed(jobs)}
        self.assertEqual(forward, backward)

    def test_render_backend_rebinds_context_before_query(self):
        from unittest.mock import patch
        bound = []
        env = NS(sim=NS(_render_context_offscreen=NS(gl_ctx=NS(make_current=lambda: bound.append(True)))))
        def query(enum):
            self.assertTrue(bound)
            return b'NVIDIA'
        with patch('OpenGL.GL.glGetString', side_effect=query):
            self.assertEqual(render_backend(env), {'vendor': 'NVIDIA', 'renderer': 'NVIDIA'})
        self.assertEqual(bound, [True])

    def test_shards_cover_same_global_episodes_without_duplicates(self):
        registry = {'a': 500, 'b': 600}
        whole = episode_specs(['a', 'b'], 16, 7, registry)
        for count in [1, 2, 8, 16]:
            shards = [episode_specs(['a', 'b'], 16, 7, registry, i, count) for i in range(count)]
            merged = [row for shard in shards for row in shard]
            self.assertEqual(len(merged), len(whole))
            self.assertEqual({(x['task'], x['seed']) for x in merged}, {(x['task'], x['seed']) for x in whole})
        with self.assertRaises(ValueError):
            episode_specs(['a'], 1, 0, registry, 2, 2)

    def test_rpc_roundtrip_and_error(self):
        import threading
        import zmq
        from evaluation.protocol import PROTOCOL
        from evaluation.transport import RemotePolicy
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        port = socket.bind_to_random_port('tcp://127.0.0.1')
        methods = []
        def serve():
            for _ in range(4):
                request = socket.recv_pyobj()
                methods.append(request['method'])
                if request['method'] == 'metadata':
                    socket.send_pyobj({'result': {'protocol': PROTOCOL}})
                elif request['method'] == 'reset':
                    socket.send_pyobj({'result': None})
                elif len(methods) == 3:
                    socket.send_pyobj({'result': np.zeros((2, 7))})
                else:
                    socket.send_pyobj({'error': 'checkpoint failure'})
        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        policy = RemotePolicy(f'tcp://127.0.0.1:{port}', timeout_ms=2000)
        try:
            policy.reset({'episode_id': 0})
            self.assertEqual(policy.predict({}).shape, (2, 7))
            with self.assertRaisesRegex(RuntimeError, 'checkpoint failure'):
                policy.predict({})
        finally:
            policy.close()
            worker.join(timeout=3)
            socket.close(linger=0)
            context.term()
        self.assertEqual(methods, ['metadata', 'reset', 'predict', 'predict'])

    def test_xwam_wire_and_no_double_gripper_flip(self):
        import pickle
        policy = XWamPolicy.__new__(XWamPolicy)
        policy.config = {}
        sent = []
        expected = np.full((4, 7), .25)
        policy.socket = NS(send=sent.append, recv=lambda: pickle.dumps({'actions': expected}))
        obs = observe(FakeEnv(), np.eye(4), 'test', 11, 0)
        np.testing.assert_array_equal(policy.predict(obs), expected)
        request = pickle.loads(sent[0])
        self.assertEqual(request['proprios'].shape, (16,))
        self.assertEqual(request['rollout_id'], 11)
        np.testing.assert_allclose(request['video'][0], obs['images'][CAMERAS[0]] / 127.5 - 1., atol=1e-6)


if __name__ == '__main__':
    unittest.main()
