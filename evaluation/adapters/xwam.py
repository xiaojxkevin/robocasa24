"""Bridge to the original X-WAM broker and policy_server, without model changes."""
import pickle
import numpy as np
import zmq
from evaluation.protocol import CAMERAS, validate_actions


class XWamPolicy:
    def __init__(self, config):
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, config.get('timeout_ms', 120000))
        self.socket.setsockopt(zmq.SNDTIMEO, config.get('timeout_ms', 120000))
        self.socket.connect(config['broker_endpoint'])

    def reset(self, episode):
        # Original server is request-local. rollout_id/step_id identify episode boundaries.
        pass

    def predict(self, obs):
        request = {
            'env_rank': self.config.get('env_rank', 0), 'rollout_id': obs['episode_id'],
            'step_id': obs['step_id'], 'prompt': [obs['instruction']],
            'cfg': self.config.get('cfg', 0.),
            'video': np.stack([obs['images'][c] for c in CAMERAS]).astype(np.float32) / 127.5 - 1.,
            'proprios': np.concatenate([obs['state8'], np.zeros(8)]),
        }
        self.socket.send(pickle.dumps(request))
        # Original policy_server already denormalizes and restores native gripper sign.
        return validate_actions(pickle.loads(self.socket.recv())['actions'])

    def close(self):
        self.socket.close()
        self.context.term()
