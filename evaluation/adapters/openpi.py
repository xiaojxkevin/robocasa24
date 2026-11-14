"""Shared by pi0, pi05, their inference-compatible variants, and Spatial Forcing."""
from pathlib import Path
import numpy as np
from evaluation.protocol import (OPENPI_NATIVE_CONTRACTS, SPATIAL_FORCING_CONTRACT, encode_openpi,
                                 validate_actions, validate_openpi_config)


class OpenPiPolicy:
    def __init__(self, config):
        from openpi.policies import policy_config
        from openpi.training import config as training_config
        validate_openpi_config(config)
        self.config = config
        train_config = training_config.get_config(config['train_config'])
        self.native_robocasa = config.get('input_contract') in OPENPI_NATIVE_CONTRACTS
        # Requests run serially on the server. This policy has no history or
        # recurrent state; only the sampling stream needs episode isolation.
        self.parallel_safe = self.native_robocasa
        kwargs = {}
        if self.native_robocasa:
            backend = config.get('backend')
            if backend == 'spatial_forcing':
                data_type = getattr(training_config, 'RoboCasaDataConfig', ())
                if (config['input_contract'] != SPATIAL_FORCING_CONTRACT
                        or not getattr(train_config, 'align_enabled', False)
                        or getattr(train_config, 'align_target_model', None) != 'vggt'):
                    raise ValueError('Spatial Forcing requires the RoboCasa VGGT alignment training config')
            else:
                data_type = training_config.RoboCasaD4rtDataConfig
            if not isinstance(train_config.data, data_type):
                raise ValueError('This input contract requires the native RoboCasa raw_action data config')
            if (backend not in ('pi0', 'pi05', 'spatial_forcing')
                    or train_config.model.pi05 != (backend in ('pi05', 'spatial_forcing'))):
                raise ValueError('backend must match the pi0/pi05 training model')
            if getattr(train_config.model, 'rtc', False):
                raise ValueError('Expected a non-RTC RoboCasa checkpoint')
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
                if backend == 'spatial_forcing':
                    for field in ('mean', 'std', 'q01', 'q99'):
                        value = np.asarray(getattr(norm_stats[key], field, None))
                        if value.shape != (dim,) or not np.isfinite(value).all():
                            raise ValueError(f'Invalid checkpoint {key}/{field} normalization')
                    if (np.any(norm_stats[key].std < 0)
                            or np.any(norm_stats[key].q01 > norm_stats[key].q99)):
                        raise ValueError(f'Invalid checkpoint {key} normalization bounds')
            kwargs['norm_stats'] = norm_stats
            steps = config.get('num_inference_steps', 10)
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
                raise ValueError('num_inference_steps must be a positive integer')
            kwargs['sample_kwargs'] = {'num_steps': steps}
            self.horizon = train_config.model.action_horizon
            self.action_dim = train_config.model.action_dim
            self.episode_rngs = {}
            config['checkpoint'] = str(checkpoint)
        self.policy = policy_config.create_trained_policy(
            train_config, config['checkpoint'], **kwargs)

    def reset(self, episode):
        if self.native_robocasa:
            # Explicit per-rollout noise makes stochastic inference reproducible
            # even when other rollouts were evaluated earlier on this server.
            key = (episode['task'], episode['episode_id'])
            self.episode_rngs[key] = np.random.default_rng(int(episode['seed']))
        if not self.native_robocasa and hasattr(self.policy, 'reset'):
            self.policy.reset()

    def predict(self, obs):
        kwargs = {}
        if self.native_robocasa:
            key = (obs['task'], obs['episode_id'])
            kwargs['noise'] = self.episode_rngs[key].standard_normal(
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
