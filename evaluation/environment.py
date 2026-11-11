"""Simulation only. Policy servers never import this module."""
import numpy as np
from scipy.spatial.transform import Rotation
from evaluation.protocol import CAMERAS, EEF_AXES


def create_env(task, seed):
    import robocasa  # noqa: F401 -- register kitchen tasks
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    return robosuite.make(
        env_name=task, robots='PandaOmron',
        controller_configs=load_composite_controller_config(controller=None, robot='PandaOmron'),
        camera_names=list(CAMERAS), camera_widths=256, camera_heights=256,
        control_freq=20, has_renderer=False, has_offscreen_renderer=True,
        ignore_done=True, use_object_obs=True, use_camera_obs=False,
        camera_depths=False, seed=seed, obj_instance_split='B',
        generative_textures=None, randomize_cameras=False,
        layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
        translucent_robot=False,
    )


def base_transform(env):
    c = env.robots[0].composite_controller
    arm = c.part_controllers[c.arms[0]]
    transform = np.eye(4)
    transform[:3, :3] = arm.origin_ori
    transform[:3, 3] = arm.origin_pos
    return transform


def observe(env, base, task, episode, step):
    c = env.robots[0].composite_controller
    arm = c.part_controllers[c.arms[0]]
    transform = np.eye(4)
    transform[:3, :3] = arm.ref_ori_mat
    transform[:3, 3] = arm.ref_pos
    relative = np.linalg.inv(base) @ transform
    rotm = relative[:3, :3] @ EEF_AXES
    # SciPy 1.10 (compatible with the pinned simulator) lacks canonical=True.
    quat = Rotation.from_matrix(rotm).as_quat()[[3, 0, 1, 2]]
    nonzero = quat[np.flatnonzero(quat)]
    if len(nonzero) and nonzero[0] < 0:
        quat = -quat
    grip = c.part_controllers[list(c.grippers)[0]]
    opening = grip.joint_pos[0:1] / grip.actuator_max[0]
    return {
        'images': {name: env.sim.render(height=256, width=256, camera_name=name,
                                      depth=False, segmentation=False)[::-1].copy() for name in CAMERAS},
        'state8': np.concatenate([relative[:3, 3], quat, opening]).astype(np.float32),
        # Preserve the original observation convention; do not force controller updates.
        'state10': np.concatenate([relative[:3, 3], rotm[:, 0], rotm[:, 1], opening]).astype(np.float32),
        'state13': np.concatenate([relative[:3, 3], rotm.reshape(-1), opening]).astype(np.float32),
        'instruction': env.get_ep_meta()['lang'], 'task': task,
        'episode_id': episode, 'step_id': step,
    }


def render_backend(env):
    """Bind this environment explicitly; cleanup of older EGL contexts may unbind it."""
    from OpenGL import GL
    env.sim._render_context_offscreen.gl_ctx.make_current()
    result = {}
    for key, enum in [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER)]:
        value = GL.glGetString(enum)
        if value is None:
            raise RuntimeError(f'GL {key} unavailable after binding the environment context')
        result[key] = value.decode()
    return result
