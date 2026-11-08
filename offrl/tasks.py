"""A Task bundles everything that differs between environments: the dataset, the test environment,
how observations become state vectors, how transitions are labeled, and how success is measured.
The algorithms, the training loops and the evaluation code are shared by all tasks."""
import minari
import numpy as np

from .data import load_maze_transitions, load_mujoco_transitions, make_state, reward_scale_from_dataset, test_goal_of
from .envs.maze import make_eval_env, maze_geometry, upright_of
from .envs.mujoco import make_mujoco_env


class Task:
    """kind = "maze" (PointMaze / AntMaze: dict observations, sparse goal reward, metric = success rate)
    or "mujoco" (locomotion: box observations, dense reward, metric = episode return)."""

    def __init__(self, cfg, verbose=True):
        self.cfg = cfg
        self.dataset = minari.load_dataset(cfg["dataset_id"], download=True)
        self.kind = cfg.get("task_type") or ("maze" if "maze" in cfg["dataset_id"] else "mujoco")
        if self.kind == "maze":
            self.fixed_goal = bool(cfg.get("relabel_to_test_goal", False))
            self.env = make_eval_env(self.dataset, verbose=verbose, fixed_goal=self.fixed_goal)
            self.geom = maze_geometry(self.env)
            self.is_ant = "ant" in self.env.spec.id.lower()
            self.metrics = ("success", "flipped") if self.is_ant else ("success",)
        else:
            self.env = make_mujoco_env(self.dataset)
            self.is_ant = False
            self.reward_scale = reward_scale_from_dataset(self.dataset) if cfg.get("normalize_reward", True) else 1.0
            # Minari's mujoco/* datasets have no reference (random / expert) returns, so there is no D4RL-style
            # normalized score. Instead: the raw return, and the return as a % of the dataset's mean episode return.
            self.data_return = float(np.mean([np.sum(ep.rewards) for ep in self.dataset.iterate_episodes()]))
            has_score = self.normalized_score(np.zeros(1)) is not None
            self.metrics = (("score",) if has_score else ()) + ("return", "pct_data")
        self.primary = self.metrics[0]
        self.curve_metrics = self.metrics if self.kind == "maze" else (self.primary,)   # what learning curves show
        self.max_steps = self.env.spec.max_episode_steps
        if verbose:
            print(f"task: {cfg['dataset_id']} ({self.kind}{', ant' if self.is_ant else ''}) | "
                  f"{self.dataset.total_episodes:,} episodes, {self.dataset.total_steps:,} steps | "
                  f"test episodes: {self.max_steps} steps | metric: {self.primary}")

    # --- environments and states ---
    def make_env(self, render_mode=None):
        if self.kind == "maze":
            return make_eval_env(self.dataset, render_mode=render_mode, verbose=False, fixed_goal=self.fixed_goal)
        return make_mujoco_env(self.dataset, render_mode=render_mode)

    def state(self, obs):
        return make_state(obs) if self.kind == "maze" else np.asarray(obs, dtype=np.float32)

    # --- offline data ---
    def transitions(self):
        c = self.cfg
        if self.kind == "mujoco":
            return load_mujoco_transitions(self.dataset, self.reward_scale)
        penalty = c.get("flip_penalty")
        return load_maze_transitions(
            self.dataset, c.get("reward_shift", -1.0),
            test_goal=test_goal_of(self.env) if self.fixed_goal else None,
            radius=c.get("goal_radius", 0.45), cut_flips=c.get("cut_flips"),
            flip_terminal=c.get("flip_terminal", False),
            flip_penalty=penalty if penalty is not None else -1.0 / (1.0 - c["discount"]))

    # --- online labels: the same as the offline data ---
    def success(self, r, info):
        return self.kind == "maze" and bool(info.get("success", False) or r > 0)

    def flipped(self, obs):
        return self.is_ant and upright_of(obs) < 0

    def online_label(self, r, terminated, info):
        """(reward, done) for a new transition, labeled exactly like the offline data."""
        if self.kind == "maze":
            ok = self.success(r, info)
            return float(ok) + self.cfg.get("reward_shift", -1.0), ok
        return float(r) * self.reward_scale, bool(terminated)

    def episode_over(self, r, terminated, truncated, info, next_obs, reset_on_flip=True):
        if self.kind == "maze":
            return self.success(r, info) or terminated or truncated or (reset_on_flip and self.flipped(next_obs))
        return terminated or truncated

    # --- scores ---
    def normalized_score(self, returns):
        """D4RL-style score (x100) from Minari's reference scores, or None if the dataset has none."""
        try:
            return np.asarray(minari.get_normalized_score(self.dataset, np.asarray(returns, dtype=np.float64))) * 100.0
        except Exception:
            return None
