"""Batched evaluation: many episodes (of all seeds) step together, with one compiled policy call
per time step for all of them. Test actions are the deterministic policy mean, as in the
official IQL code and CORL."""
import jax
import numpy as np

from .envs.maze import xy_to_cell
from .networks import policy_mean

# actions for every seed at once: params have a leading seed axis, states are shared
act_all_seeds = jax.jit(jax.vmap(policy_mean, in_axes=(0, None)))


class EnvPool:
    """Reusable test environments (creating some environments is slow)."""
    def __init__(self, task):
        self.task, self.envs = task, []

    def get(self, n):
        while len(self.envs) < n:
            self.envs.append(self.task.make_env())
        return self.envs[:n]


STUCK_WINDOW = 100        # steps
STUCK_SPEED = 0.5         # m/s: below this average forward speed over the last STUCK_WINDOW steps = "stuck"


def run_episodes(task, pi_params, specs, pool, normalizer, keep_paths=False, max_parallel=150):
    """specs: list of (seed_index, env_seed, reset_options).
    Returns dict of per-episode arrays: success, ret (undiscounted env return), length, flipped,
    and (maze, keep_paths=True) the xy path of each episode.
    MuJoCo also: fell (the episode terminated early) and stuck (it ran to the time limit but its average forward
    speed over the last STUCK_WINDOW steps was below STUCK_SPEED, e.g. a HalfCheetah lying on its back)."""
    n = len(specs)
    out = dict(success=np.zeros(n, bool), ret=np.zeros(n), length=np.zeros(n, int), flipped=np.zeros(n, bool))
    mujoco = task.kind != "maze"
    if mujoco:
        out.update(fell=np.zeros(n, bool), stuck=np.zeros(n, bool))
        dt = task.env.unwrapped.dt
    paths = [None] * n
    for start in range(0, n, max_parallel):
        chunk = specs[start:start + max_parallel]
        envs = pool.get(len(chunk))
        seed_idx = np.array([c[0] for c in chunk])
        obs = [env.reset(seed=int(es), options=ro)[0] for env, (_, es, ro) in zip(envs, chunk)]
        pth = [[o["achieved_goal"].copy()] for o in obs] if keep_paths and task.kind == "maze" else None
        xs = [[float(env.unwrapped.data.qpos[0])] for env in envs] if mujoco else None
        active = np.ones(len(chunk), bool)
        while active.any():
            S = normalizer(np.stack([task.state(o) for o in obs]))
            A = np.asarray(act_all_seeds(pi_params, S))[seed_idx, np.arange(len(chunk))]
            for i in np.flatnonzero(active):
                obs[i], r, term, trunc, info = envs[i].step(A[i])
                k = start + i
                out["ret"][k] += r; out["length"][k] += 1
                if pth is not None:
                    pth[i].append(obs[i]["achieved_goal"].copy())
                if xs is not None:
                    xs[i].append(float(info.get("x_position", envs[i].unwrapped.data.qpos[0])))
                if task.success(r, info):
                    out["success"][k] = True; active[i] = False
                elif term or trunc:
                    active[i] = False; out["flipped"][k] = task.flipped(obs[i])
                    if mujoco:
                        out["fell"][k] = bool(term)
                        w = min(STUCK_WINDOW, len(xs[i]) - 1)
                        speed = (xs[i][-1] - xs[i][-1 - w]) / (w * dt) if w > 0 else 0.0
                        out["stuck"][k] = (not term) and speed < STUCK_SPEED
        if pth is not None:
            for i in range(len(chunk)):
                paths[start + i] = np.array(pth[i])
    if keep_paths:
        out["paths"] = paths
    return out


def format_metrics(ev):
    """One log line for a dict of per-seed metric arrays, e.g. 'return [14817 12815] (mean 13816) | pct_data ...'."""
    parts = []
    for k, v in ev.items():
        v = np.asarray(v, dtype=float)
        d = 0 if np.abs(v).max() >= 1000 else (1 if np.abs(v).max() >= 10 else 2)
        vals = " ".join(f"{x:.{d}f}" for x in v)
        parts.append(f"{k} [{vals}] (mean {v.mean():.{d}f})")
    return " | ".join(parts)


def evaluate(task, pi_params, n_seeds, n_episodes, seed0, pool, normalizer, return_episodes=False):
    """Per-seed metrics: maze -> success (and flipped, for AntMaze); mujoco -> return and pct_data
    (return as % of the dataset's mean episode return), plus score if the dataset has reference returns."""
    specs = [(i, seed0 + e, None) for i in range(n_seeds) for e in range(n_episodes)]
    out = run_episodes(task, pi_params, specs, pool, normalizer)
    per_seed = lambda x: x.reshape(n_seeds, n_episodes).mean(1)
    if task.kind == "maze":
        res = {"success": per_seed(out["success"].astype(float))}
        if task.is_ant:
            res["flipped"] = per_seed(out["flipped"].astype(float))
    else:
        ret = per_seed(out["ret"])
        res = {"return": ret, "pct_data": 100.0 * ret / task.data_return}
        if "score" in task.metrics:
            res = {"score": task.normalized_score(ret), **res}
    if return_episodes:                     # per-episode outcomes, (n_seeds, n_episodes) each
        keys = ("ret", "length", "success", "flipped") + (("fell", "stuck") if task.kind != "maze" else ())
        eps = {k: out[k].reshape(n_seeds, n_episodes) for k in keys}
        return res, eps
    return res


def success_maps(task, pi_params, n_seeds, pool, normalizer, episodes_per_cell=3, seed=20000):
    """Maze only: success rate from every free start cell, per seed. (n_seeds, rows, cols), NaN on walls/goal."""
    geom, env = task.geom, task.env
    walls = np.array(geom["walls"])
    goal_cell = xy_to_cell(geom, env.reset(seed=seed)[0]["desired_goal"])
    cells = [(i, j) for i in range(walls.shape[0]) for j in range(walls.shape[1])
             if not walls[i, j] and (i, j) != goal_cell]
    specs = [(si, seed + episodes_per_cell * k + e, {"reset_cell": np.array(c)})
             for si in range(n_seeds) for k, c in enumerate(cells) for e in range(episodes_per_cell)]
    ok = run_episodes(task, pi_params, specs, pool, normalizer)["success"]
    ok = ok.reshape(n_seeds, len(cells), episodes_per_cell).mean(2)
    maps = np.full((n_seeds,) + walls.shape, np.nan)
    for k, (i, j) in enumerate(cells):
        maps[:, i, j] = ok[:, k]
    return maps


def evaluate_final(task, cfg, states_by_method, logs_by_method, pool, normalizer, n_episodes=None, maps=True):
    """Final test for several methods (e.g. {"BC": ..., "IQL": ...}); JSON-friendly result dict."""
    n = len(cfg["seeds"])
    n_episodes = n_episodes or cfg["final_episodes"]
    res = dict(dataset_id=cfg["dataset_id"], kind=task.kind, metrics=list(task.metrics), seeds=list(cfg["seeds"]),
               n_episodes=n_episodes, methods={})
    if task.kind == "maze":
        res.update(geom=task.geom, goal=task.env.reset(seed=20000)[0]["desired_goal"].tolist(),
                   episodes_per_cell=cfg["episodes_per_cell"])
    else:
        res.update(data_return=task.data_return)
    res["max_steps"] = task.max_steps
    for name, st in states_by_method.items():
        ev, eps = evaluate(task, st["pi"], n, n_episodes, 5000, pool, normalizer, return_episodes=True)
        m = success_maps(task, st["pi"], n, pool, normalizer, cfg["episodes_per_cell"]) \
            if (maps and task.kind == "maze") else None
        log = (logs_by_method or {}).get(name)
        res["methods"][name] = [
            dict(seed=s, final={k: float(v[i]) for k, v in ev.items()},
                 episodes={k: np.asarray(v[i]).tolist() for k, v in eps.items()},
                 map=m[i].tolist() if m is not None else None,
                 log=dict(step=log["step"], **{k: [x[i] for x in log[k]] for k in task.metrics if k in log})
                 if log else None)
            for i, s in enumerate(cfg["seeds"])]
    return res


def summarize(res):
    """Prints mean +- std over seeds of every final metric (and the start-cell average for mazes;
    for MuJoCo also the fraction of episodes that fell or got stuck)."""
    methods = res["methods"]
    first = next(iter(methods.values()))[0]
    cols = res["metrics"] + (["all start cells"] if res["kind"] == "maze" and first["map"] is not None else [])
    ep_cols = [c for c in ("fell", "stuck") if res["kind"] != "maze" and c in (first.get("episodes") or {})]
    cols = cols + ep_cols
    print(f"{res['dataset_id']}  ({len(res['seeds'])} seeds, {res['n_episodes']} test episodes per seed)")
    print(f"{'method':18s}" + "".join(f"{c:>21s}" for c in cols))
    for name, runs in methods.items():
        line = f"{name:18s}"
        for c in cols:
            v = np.array([np.nanmean(np.array(r["map"], dtype=float)) if c == "all start cells" else
                          np.mean(r["episodes"][c]) if c in ep_cols else r["final"][c] for r in runs])
            d = 0 if np.abs(v).max() >= 1000 else (1 if np.abs(v).max() >= 10 else 2)
            line += f"{v.mean():12,.{d}f} ± {v.std():<6,.{d}f}"
        print(line)
