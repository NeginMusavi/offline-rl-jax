"""Maze tasks (PointMaze, AntMaze from Minari): the test environment and maze geometry.

Two fixes relative to Minari's defaults live here:
- the evaluation maze stored with some datasets is rotated/flipped relative to the maze the data was
  recorded in (seen on D4RL/antmaze/umaze); make_eval_env finds the orientation that matches the data;
- FixedGoal puts the test goal exactly at the goal-cell center, matching relabeled rewards
  (Gymnasium otherwise moves it by up to +-1 on every reset).
"""
import gymnasium as gym
import numpy as np


def make_eval_env(dataset, render_mode=None, verbose=True, fixed_goal=False):
    # Minari's eval env can have its maze map rotated/flipped relative to the maze the data was
    # recorded in (seen on D4RL/antmaze/umaze). Fix: take the eval map (it holds the fixed start "r"
    # and goal "g"), find the rotation/flip whose walls match the data env, then check the data.
    data_env = dataset.recover_environment()
    given = dataset.recover_environment(eval_env=True)
    walls_of = lambda m: np.array([[cell == 1 for cell in row] for row in m])
    target = walls_of(data_env.unwrapped.maze.maze_map)
    m = [list(row) for row in given.unwrapped.maze.maze_map]
    candidates = [("as given", m),
                  ("rotated 180°", [row[::-1] for row in m[::-1]]),
                  ("mirrored left-right", [row[::-1] for row in m]),
                  ("mirrored top-bottom", m[::-1])]
    for fix, maze_map in candidates:
        w = walls_of(maze_map)
        if w.shape == target.shape and (w == target).all():
            break
    else:
        raise RuntimeError("Eval maze doesn't match the data maze under any rotation/flip")
    kwargs = {k: v for k, v in given.spec.kwargs.items() if k != "render_mode"}
    kwargs["maze_map"] = maze_map
    env = gym.make(given.spec.id, max_episode_steps=given.spec.max_episode_steps, render_mode=render_mode, **kwargs)
    data_env.close(); given.close()

    # Sanity check: dataset positions should (almost) never be inside a wall of the eval maze
    mz = env.unwrapped.maze
    xy = np.concatenate([e.observations["achieved_goal"] for e in dataset.sample_episodes(min(50, dataset.total_episodes))])
    rows = ((mz.y_map_center - xy[:, 1]) // mz.maze_size_scaling).astype(int).clip(0, target.shape[0] - 1)
    cols = ((xy[:, 0] + mz.x_map_center) // mz.maze_size_scaling).astype(int).clip(0, target.shape[1] - 1)
    in_walls = target[rows, cols].mean()
    if verbose:
        print(f"Eval maze: {fix} | dataset positions inside walls: {in_walls:.2%}")
        for row in maze_map:
            print("   ", row)
    assert in_walls < 0.02, "Eval maze still doesn't match the data"
    if fixed_goal:   # put the test goal exactly at the center of the "g" cell (see Section 3)
        (gi, gj), = [(i, j) for i, row in enumerate(maze_map) for j, cell in enumerate(row) if cell == "g"]
        env = FixedGoal(env, mz.cell_rowcol_to_xy(np.array([gi, gj])))
        if verbose:
            print("Test goal fixed at the goal-cell center:", env.goal)
    return env

class FixedGoal(gym.Wrapper):
    # Gymnasium moves the goal randomly by up to ±1 around the goal cell's center on every reset.
    # This wrapper keeps it at the center, matching the relabeled rewards.
    def __init__(self, env, goal):
        super().__init__(env)
        self.goal = np.asarray(goal, dtype=np.float64)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        u = self.env.unwrapped
        u.goal = self.goal.copy()          # the goal the env uses for reward / success
        u.update_target_site_pos()         # move the red ball in the video
        obs = dict(obs); obs["desired_goal"] = self.goal.copy()
        return obs, info


def maze_geometry(env):
    # Everything needed to draw the maze, stored as plain Python so it can be saved to JSON
    m = env.unwrapped.maze
    walls = [[cell == 1 for cell in row] for row in m.maze_map]   # cells can also be "g"/"r" (free)
    return dict(walls=walls, x_c=float(m.x_map_center), y_c=float(m.y_map_center), scale=float(m.maze_size_scaling))

def maze_extent(g):
    return [-g["x_c"], g["x_c"], -g["y_c"], g["y_c"]]

def cell_center(g, i, j):
    return (j + 0.5) * g["scale"] - g["x_c"], g["y_c"] - (i + 0.5) * g["scale"]

def xy_to_cell(g, xy):
    return int((g["y_c"] - xy[1]) // g["scale"]), int((xy[0] + g["x_c"]) // g["scale"])

def draw_maze(ax, g):
    # ax: a matplotlib axis
    ax.imshow(np.array(g["walls"]), cmap="Greys", extent=maze_extent(g), origin="upper", alpha=0.8, zorder=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")


def upright_of(obs):
    """AntMaze only: 1 = torso upright, 0 = on its side, -1 = upside down."""
    w, x, y, z = obs["observation"][1:5]
    return 1 - 2 * (x ** 2 + y ** 2)
