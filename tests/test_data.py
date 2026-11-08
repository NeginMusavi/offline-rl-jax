"""Data labeling on small hand-made episodes (no dataset download needed)."""
from types import SimpleNamespace

import numpy as np

from offrl.data import Normalizer, load_maze_transitions, load_mujoco_transitions, reward_scale_from_dataset


class FakeDataset:
    def __init__(self, episodes):
        self.episodes = episodes

    def iterate_episodes(self):
        return iter(self.episodes)


def maze_episode(xy, upside_down_from=None, own_goal=(0.0, 0.0)):
    """A fake AntMaze episode along the given xy path; the ant is upside down from a given step on."""
    T1 = len(xy)
    body = np.zeros((T1, 27), np.float32)
    body[:, 1] = 1.0                                           # quaternion w = 1: upright
    if upside_down_from is not None:
        body[upside_down_from:, 1] = 0.0; body[upside_down_from:, 2] = 1.0   # x = 1: upside down
    obs = dict(observation=body, achieved_goal=np.asarray(xy, np.float32),
               desired_goal=np.tile(np.asarray(own_goal, np.float32), (T1, 1)))
    return SimpleNamespace(observations=obs, actions=np.zeros((T1 - 1, 8), np.float32),
                           rewards=np.zeros(T1 - 1, np.float32), terminations=np.zeros(T1 - 1, bool))


def test_relabel_rewards_for_test_goal():
    xy = [(0, 0), (1, 0), (2, 0), (2.2, 0), (3, 0)]
    d = load_maze_transitions(FakeDataset([maze_episode(xy)]), -1.0, test_goal=np.array([2.0, 0.0]), radius=0.45)
    np.testing.assert_array_equal(d["dones"], [0, 1, 1, 0])     # s' = (2,0) and (2.2,0) are within 0.45
    np.testing.assert_array_equal(d["rewards"], [-1, 0, 0, -1])
    assert np.allclose(d["states"][:, -2:], [2.0, 0.0])         # goal in every state = the test goal


def test_cut_flips_and_flip_terminal():
    ep = maze_episode([(i, 0) for i in range(10)], upside_down_from=6)
    cut = load_maze_transitions(FakeDataset([ep]), -1.0, test_goal=np.array([50.0, 0.0]), cut_flips=2)
    assert len(cut["states"]) == 6 - 2 - 1                      # transitions whose s' is before the cut
    term = load_maze_transitions(FakeDataset([ep]), -1.0, test_goal=np.array([50.0, 0.0]),
                                 flip_terminal=True, flip_penalty=-100.0)
    assert len(term["states"]) == 6                             # up to and including the step into the flip
    assert term["dones"][-1] == 1 and term["rewards"][-1] == -100.0 and term["dones"][:-1].sum() == 0


def test_mujoco_transitions_and_reward_scale():
    eps = [SimpleNamespace(observations=np.arange(4 * 3, dtype=np.float32).reshape(4, 3),
                           actions=np.zeros((3, 2), np.float32), rewards=np.array(r, np.float32),
                           terminations=np.array([0, 0, t], bool))
           for r, t in [([1, 1, 1], 0), ([2, 3, 5], 1)]]
    ds = FakeDataset(eps)
    scale = reward_scale_from_dataset(ds)
    assert np.isclose(scale, 1000.0 / (10 - 3))
    d = load_mujoco_transitions(ds, scale)
    assert d["states"].shape == (6, 3) and np.allclose(d["next_states"][0], d["states"][1])
    np.testing.assert_array_equal(d["dones"], [0, 0, 0, 0, 0, 1])
    assert np.isclose(d["rewards"].sum(), 13 * scale)


def test_normalizer_round_trip():
    x = np.random.default_rng(0).normal(3.0, 2.0, size=(1000, 4)).astype(np.float32)
    n = Normalizer(x)
    assert np.abs(n(x).mean(0)).max() < 1e-4 and np.abs(n(x).std(0) - 1).max() < 1e-2
    n2 = Normalizer.from_state_dict(n.state_dict())
    np.testing.assert_array_equal(n(x), n2(x))
