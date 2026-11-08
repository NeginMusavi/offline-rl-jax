"""Turning Minari datasets into training transitions (s, a, r, s', done) and normalizing states.

Maze tasks use dict observations and goal-reaching labels; MuJoCo locomotion tasks use the
environment's dense reward, scaled as in the official IQL code.
"""
import numpy as np
import jax.numpy as jnp


# --- maze tasks -------------------------------------------------------------------------------

def make_state(obs):
    return np.concatenate([obs["achieved_goal"], obs["observation"], obs["desired_goal"]], axis=-1).astype(np.float32)

def load_maze_transitions(dataset, reward_shift=-1.0, test_goal=None, radius=0.45, cut_flips=None,
                     flip_terminal=False, flip_penalty=-100.0):
    # test_goal=None: keep Minari's rewards (each episode's own goal).
    # test_goal=(x, y): relabel every transition for the test goal (D4RL-style, see the text above).
    # cut_flips=None: keep all data. cut_flips=K: end each episode K steps before the ant first flips over.
    # flip_terminal=True: keep everything up to the first flip, make the step INTO the flip terminal with
    #                     reward flip_penalty, and drop the upside-down rest of the episode.
    assert not (cut_flips is not None and flip_terminal), "use either cut_flips or flip_terminal, not both"
    S, A, R, S2, D, FL = [], [], [], [], [], []
    n_cut = n_cut_eps = 0
    for ep in dataset.iterate_episodes():
        obs = ep.observations
        if test_goal is None:
            rew = ep.rewards
            done = np.logical_or(ep.terminations, ep.rewards > 0)       # goal reached = terminal
        else:
            obs = dict(obs)
            obs["desired_goal"] = np.broadcast_to(np.asarray(test_goal, dtype=np.float32), obs["desired_goal"].shape)
            dist = np.linalg.norm(ep.observations["achieved_goal"][1:] - np.asarray(test_goal), axis=-1)
            rew = (dist <= radius).astype(np.float32)                    # 1 when s' is at the TEST goal
            done = rew > 0
        s = make_state(obs)
        T = len(ep.actions)                                             # transitions kept from this episode
        rew, done = np.array(rew, dtype=np.float32), np.array(done, dtype=bool)
        flip = np.zeros(T, bool)
        if cut_flips is not None or flip_terminal:
            body = ep.observations["observation"]
            upside_down = np.flatnonzero(1 - 2 * (body[:, 2] ** 2 + body[:, 3] ** 2) < 0)
            if len(upside_down):
                f = int(upside_down[0])                                 # first upside-down observation
                T_new = max(0, f - cut_flips - 1) if cut_flips is not None else f
                n_cut += T - T_new; n_cut_eps += 1; T = T_new
                if flip_terminal and T > 0:                             # transition T-1 goes INTO the flip
                    done[T - 1] = True; flip[T - 1] = True
        if T == 0:
            continue
        S.append(s[:T]); S2.append(s[1:T + 1]); A.append(ep.actions[:T])
        R.append(rew[:T]); D.append(done[:T]); FL.append(flip[:T])
    if cut_flips is not None:
        print(f"cut_flips={cut_flips}: removed {n_cut:,} transitions from {n_cut_eps:,} episodes where the ant flips")
    if flip_terminal:
        print(f"flip_terminal: {n_cut_eps:,} episodes end at the flip (reward {flip_penalty:.1f}); "
              f"removed {n_cut:,} upside-down transitions")
    cat = lambda xs: np.concatenate(xs).astype(np.float32)
    data = dict(states=cat(S), actions=cat(A), rewards=cat(R), next_states=cat(S2), dones=cat(D))
    label = "own episode goals" if test_goal is None else f"relabeled for test goal {np.round(test_goal, 2)}"
    n_goal = int(data["dones"].sum()) - (int(np.concatenate(FL).sum()) if FL else 0)   # flips aren't goals
    print(f"{len(data['states']):,} transitions | rewards: {label} | "
          f"goal reached in {n_goal:,} of them ({n_goal / len(data['states']):.2%})")
    data["rewards"] = data["rewards"] + reward_shift
    if flip_terminal:
        is_flip = np.concatenate(FL)
        data["rewards"][is_flip] = flip_penalty                          # the flip's reward overrides everything
    return data

def test_goal_of(env):
    return env.reset(seed=0)[0]["desired_goal"]


# --- MuJoCo locomotion tasks -----------------------------------------------------------------

def reward_scale_from_dataset(dataset):
    """Official IQL normalization for locomotion: rewards * 1000 / (max episode return - min episode return)."""
    returns = np.array([float(np.sum(ep.rewards)) for ep in dataset.iterate_episodes()])
    span = returns.max() - returns.min()
    return 1000.0 / span if span > 0 else 1.0


def load_mujoco_transitions(dataset, reward_scale=1.0):
    """Box observations; done = the environment's termination (e.g. Hopper falling), not time limits."""
    S, A, R, S2, D = [], [], [], [], []
    for ep in dataset.iterate_episodes():
        obs = np.asarray(ep.observations, dtype=np.float32)
        S.append(obs[:-1]); S2.append(obs[1:]); A.append(ep.actions)
        R.append(np.asarray(ep.rewards) * reward_scale); D.append(ep.terminations)
    cat = lambda xs: np.concatenate(xs).astype(np.float32)
    data = dict(states=cat(S), actions=cat(A), rewards=cat(R), next_states=cat(S2), dones=cat(D))
    print(f"{len(data['states']):,} transitions | reward scale {reward_scale:.4f} | "
          f"terminal in {int(data['dones'].sum()):,} of them ({data['dones'].mean():.2%})")
    return data


# --- shared ----------------------------------------------------------------------------------

class Normalizer:
    """Zero-mean, unit-variance states (statistics from the offline data, reused online)."""
    def __init__(self, x=None, mean=None, std=None):
        if x is not None:
            mean, std = x.mean(0), x.std(0) + 1e-3
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    def __call__(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d):
        return cls(mean=d["mean"], std=d["std"])


def to_device(data, normalizer):
    """Normalized GPU arrays in the layout the trainers use: s, a, r, s2, d.
    Pass the result to jitted functions as an argument (as a global, jit would copy it into the program)."""
    return dict(s=jnp.asarray(normalizer(data["states"])), s2=jnp.asarray(normalizer(data["next_states"])),
                a=jnp.asarray(data["actions"]), r=jnp.asarray(data["rewards"]), d=jnp.asarray(data["dones"]))
