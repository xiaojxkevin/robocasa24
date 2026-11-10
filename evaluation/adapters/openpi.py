"""Shared by pi0, pi05, their inference-compatible variants, and Spatial Forcing."""
import numpy as np
from evaluation.protocol import encode_openpi, validate_actions


class OpenPiPolicy:
    def __init__(self, config):
        from openpi.policies import policy_config
        from openpi.training import config as training_config
        self.config = config
        self.policy = policy_config.create_trained_policy(
            training_config.get_config(config['train_config']), config['checkpoint'])

    def reset(self, episode):
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def predict(self, obs):
        actions = np.asarray(self.policy.infer(encode_openpi(obs, self.config))['actions'])
        if actions.ndim != 2:
            raise ValueError(f'Expected [T,D] policy output, got {actions.shape}')
        # Padding removal must be declared; never silently interpret joint targets as raw actions.
        indices = self.config.get('action_indices', list(range(7)))
        if len(indices) != 7 or len(set(indices)) != 7:
            raise ValueError('action_indices must select exactly seven distinct native control dimensions')
        result = actions[:, indices].copy()
        result[:, 6] *= self.config.get('gripper_sign', 1)
        return validate_actions(result)
