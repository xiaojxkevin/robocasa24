"""Reuse FastWAM model loading/normalization; no RoboTwin simulator dependencies."""
import importlib.util
import inspect
from pathlib import Path
import numpy as np
from evaluation.protocol import CAMERAS, validate_actions


class FastWamPolicy:
    def __init__(self, config):
        import torch
        from omegaconf import OmegaConf
        self.config = config
        source = Path(config['project']) / 'experiments/robotwin/fastwam_policy/deploy_policy.py'
        spec = importlib.util.spec_from_file_location('robocasa24_fastwam_runtime', source)
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        cfg = OmegaConf.load(config['model_config'])  # resolved RoboCasa training config, not Hydra defaults
        self.engine = runtime.WorldActionRobotWinPolicy(
            model_cfg=cfg.model, processor_cfg=cfg.data.train.processor,
            checkpoint_path=config['checkpoint'], dataset_stats_path=Path(config['dataset_stats']),
            device=config.get('device', 'cuda'),
            model_dtype=runtime._mixed_precision_to_model_dtype(config.get('mixed_precision', 'bf16')),
            action_horizon=config['action_horizon'], replan_steps=1,
            num_inference_steps=config.get('num_inference_steps', 4),
            sigma_shift=config.get('sigma_shift'), seed=config.get('seed', 0),
            text_cfg_scale=config.get('text_cfg_scale', 1.), negative_prompt='',
            rand_device='cpu', tiled=False, timing_enabled=False,
            num_video_frames=config['num_video_frames'],
        )
        self.runtime = runtime
        self.torch = torch

    def reset(self, episode):
        self.engine.reset()

    def predict(self, obs):
        from PIL import Image
        engine = self.engine
        # Explicit montage contract shared with the future RoboCasa training conversion:
        # left external above, right external / wrist below (320x384).
        tiles = [np.asarray(Image.fromarray(obs['images'][c]).resize(size))
                 for c, size in zip(CAMERAS, [(320, 256), (160, 128), (160, 128)])]
        image = np.concatenate([tiles[0], np.concatenate(tiles[1:], axis=1)], axis=0)
        image = self.torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=engine.model.device, dtype=engine.model.torch_dtype) * (2. / 255.) - 1.
        kwargs = dict(
            prompt=self.runtime.DEFAULT_PROMPT.format(task=obs['instruction']), input_image=image,
            proprio=engine._normalize_state(obs[self.config['state_representation']]),
            action_horizon=engine.action_horizon, negative_prompt=engine.negative_prompt,
            text_cfg_scale=engine.text_cfg_scale, num_inference_steps=engine.num_inference_steps,
            sigma_shift=engine.sigma_shift, seed=engine.seed, rand_device=engine.rand_device,
            tiled=engine.tiled,
        )
        if 'num_video_frames' in inspect.signature(engine.model.infer_action).parameters:
            kwargs['num_video_frames'] = engine._num_video_frames
        with self.torch.no_grad():
            prediction = engine.model.infer_action(**kwargs)
        actions = validate_actions(engine._denormalize_action(prediction['action'])[0]).copy()
        actions[:, 6] *= self.config.get('gripper_sign', 1)
        return validate_actions(actions)
