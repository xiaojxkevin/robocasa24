"""Shared by pi0, pi05, their inference-compatible variants, and Spatial Forcing."""
from pathlib import Path
import numpy as np
from evaluation.protocol import (OPENPI_ROBOCASA_CONTRACT, encode_openpi,
                                 validate_actions, validate_openpi_config)


class OpenPiPolicy:
    def __init__(self, config):
        from openpi.policies import policy_config
        from openpi.training import config as training_config
        validate_openpi_config(config)
        self.config = config
        train_config = training_config.get_config(config['train_config'])
        self.native_robocasa = config.get('input_contract') == OPENPI_ROBOCASA_CONTRACT
        kwargs = {}
        if self.native_robocasa:
            if not isinstance(train_config.data, training_config.RoboCasaD4rtDataConfig):
                raise ValueError('This input contract requires the native RoboCasa raw_action data config')
            if not train_config.model.pi05 or train_config.model.rtc:
                raise ValueError('Expected the non-RTC pi05 RoboCasa checkpoint')
            if train_config.model.action_horizon != config.get('action_horizon'):
                raise ValueError('Configured action_horizon differs from training')
            # Verify checkpoint-local stats before allocating the model on the GPU.
            # Do not substitute the workspace stats or base-model normalization.
            from openpi.training.checkpoints import load_norm_stats
            checkpoint = Path(config['checkpoint']).expanduser().resolve()
            asset_id = train_config.data.assets.asset_id or train_config.data.repo_id
            norm_stats = load_norm_stats(checkpoint / 'assets', asset_id)
            for key, dim in [('state', 10), ('actions', 7)]:
                if key not in norm_stats or np.asarray(norm_stats[key].mean).shape != (dim,):
                    raise ValueError(f'Checkpoint normalization requires {key} dimension {dim}')
            kwargs['norm_stats'] = norm_stats
            steps = config.get('num_inference_steps', 10)
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
                raise ValueError('num_inference_steps must be a positive integer')
            kwargs['sample_kwargs'] = {'num_steps': steps}
            self.horizon = train_config.model.action_horizon
            self.action_dim = train_config.model.action_dim
            self.noise_rng = np.random.default_rng(0)
            config['checkpoint'] = str(checkpoint)
        self.policy = policy_config.create_trained_policy(
            train_config, config['checkpoint'], **kwargs)

    def reset(self, episode):
        if self.native_robocasa:
            # Explicit per-rollout noise makes stochastic inference reproducible
            # even when other rollouts were evaluated earlier on this server.
            self.noise_rng = np.random.default_rng(int(episode['seed']))
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def predict(self, obs):
        kwargs = {}
        if self.native_robocasa:
            kwargs['noise'] = self.noise_rng.standard_normal(
                (self.horizon, self.action_dim)).astype(np.float32)
        actions = np.asarray(self.policy.infer(encode_openpi(obs, self.config), **kwargs)['actions'])
        if actions.ndim != 2:
            raise ValueError(f'Expected [T,D] policy output, got {actions.shape}')
        if self.native_robocasa and actions.shape != (self.horizon, 7):
            raise ValueError(f'Expected native [{self.horizon},7] controller commands, got {actions.shape}')
        # Public policy outputs have already been unnormalized. Do not apply
        # another delta transform, orientation conversion or sign change.
        indices = self.config.get('action_indices', list(range(7)))
        if (len(indices) != 7 or len(set(indices)) != 7 or
                any(not isinstance(i, int) or i < 0 or i >= actions.shape[1] for i in indices)):
            raise ValueError('action_indices must select exactly seven distinct valid control dimensions')
        result = actions[:, indices].copy()
        result[:, 6] *= self.config.get('gripper_sign', 1)
        return validate_actions(result)
