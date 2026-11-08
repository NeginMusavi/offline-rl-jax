"""MuJoCo locomotion tasks (HalfCheetah, Hopper, ... from Minari's mujoco/* datasets)."""


def make_mujoco_env(dataset, render_mode=None):
    """The environment the dataset was recorded in (Gymnasium, with its 1000-step time limit)."""
    return dataset.recover_environment(render_mode=render_mode)
