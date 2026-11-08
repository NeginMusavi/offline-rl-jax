"""Figures. Every function accepts save_path=... to also write a PNG (e.g. into results/<name>/figures)."""
import os
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FuncFormatter

from .envs.maze import cell_center, draw_maze, maze_extent
from .evaluation import act_all_seeds, run_episodes
from .networks import value

COLORS = {"BC": "tab:blue", "IQL": "tab:orange", "offline IQL": "tab:orange", "fine-tuned IQL": "tab:green"}
LABELS = {"success": "success rate", "score": "normalized score", "return": "episode return",
          "pct_data": "% of the dataset's mean return", "flipped": "fraction flipped"}


def _fmt_pm(m, s, bounded):
    """'mean ± std' with the same number of decimals for both (set by the mean), e.g. 3,551 ± 22 or 0.99 ± 0.01."""
    if bounded or abs(m) < 10:
        return f"{m:.2f} ± {s:.2f}"
    if abs(m) < 1000:
        return f"{m:.1f} ± {s:.1f}"
    return f"{m:,.0f} ± {s:,.0f}"


def _fmt(v, bounded):
    """Rates with 2 decimals, large values (returns) without decimals and with thousands separators."""
    return f"{v:.2f}" if bounded or abs(v) < 10 else (f"{v:.1f}" if abs(v) < 1000 else f"{v:,.0f}")


def _finish(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def _steps_axis(ax):
    """Tick labels like 200k / 1.5M instead of matplotlib's 1e6 offset."""
    fmt = lambda v, _: f"{v / 1e6:g}M" if abs(v) >= 1e6 else (f"{v / 1e3:g}k" if abs(v) >= 1e3 else f"{v:g}")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))


def _nanmean(x, axis=None):
    with warnings.catch_warnings():                 # wall / goal cells are NaN in every seed
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _final_bars(ax, values_by_name, ylabel, bounded, show_seeds=False):
    """Final test: one bar per method at the mean over seeds, and an error bar from mean - std to mean + std
    (std over seeds), with the value 'mean ± std' above it. show_seeds=True also draws one dot per seed."""
    names = list(values_by_name)
    vals = [np.asarray(values_by_name[n], dtype=float) for n in names]
    draw_mean_std_bars(ax, names, [v.mean() for v in vals], [v.std() for v in vals], ylabel, bounded,
                       seeds=vals if show_seeds else None)


def draw_mean_std_bars(ax, names, means, stds, ylabel, bounded, seeds=None):
    """Bars at `means` with error bars from mean - std to mean + std. `seeds` (optional): per-method arrays of
    the individual seed values, drawn as dots."""
    tops = []
    for i, (name, m, s) in enumerate(zip(names, means, stds)):
        ax.bar(i, m, width=0.6, color=COLORS.get(name, f"C{i}"), alpha=0.75, zorder=1)
        ax.errorbar(i, m, yerr=s, fmt="none", ecolor="k", elinewidth=1.6, capsize=8, capthick=1.6, zorder=4)
        top = m + s
        if seeds is not None:
            f = np.asarray(seeds[i], dtype=float)
            ax.scatter(i + np.linspace(-0.18, 0.18, len(f)), f, s=26, color="white", edgecolor="k", zorder=3)
            top = max(top, f.max())
        tops.append(top)
    span = 1.0 if bounded else max(max(tops), 1e-9)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, tops[i] + 0.03 * span, _fmt_pm(m, s, bounded), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    if bounded:
        ax.set_ylim(0, 1.15); ax.set_yticks(np.linspace(0, 1, 6))
    else:
        ax.set_ylim(min(0.0, min(m - s for m, s in zip(means, stds))), max(tops) + 0.15 * span)
        _values_axis(ax)


def _values_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}"))


def _reference_lines(ax, refs):
    """refs: {label: value} drawn as dashed horizontal lines (e.g. the dataset's mean return)."""
    for (label, v), c in zip((refs or {}).items(), ["0.35", "tab:blue", "tab:purple"]):
        ax.axhline(v, ls="--", lw=1.2, color=c, label=label, zorder=0)


def _band(ax, x, Y, color, label):
    """Mean over seeds (line with markers) and a shaded mean +- std band. The band is clipped to the range the
    seeds actually reached, so it never shows values no seed had (e.g. above Hopper's ~3,600 ceiling)."""
    Y = np.asarray(Y, dtype=float)                 # (n_seeds, n_points)
    m, s = Y.mean(0), Y.std(0)
    lo, hi = np.maximum(m - s, Y.min(0)), np.minimum(m + s, Y.max(0))
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0)
    ax.plot(x, m, "o-", color=color, lw=2, ms=4, label=label)


def plot_offline_results(res, save_path=None):
    """Learning curves (mean +- std over seeds) and final test, for every method in an evaluate_final result.
    Without training logs (e.g. offline vs. fine-tuned) only the final-test panel is drawn."""
    if res["kind"] != "maze":
        return _mj_plot_offline_results(res, save_path)
    metric = res["metrics"][0]
    names = list(res["methods"])
    has_curves = any(runs[0]["log"] and runs[0]["log"].get(metric) for runs in res["methods"].values())
    fig, axes = plt.subplots(1, 2 if has_curves else 1, figsize=(12 if has_curves else 6, 4))
    bar_ax = axes[1] if has_curves else axes
    if has_curves:
        ax = axes[0]
        for i, name in enumerate(names):
            runs = res["methods"][name]
            if runs[0]["log"] and runs[0]["log"].get(metric):
                _band(ax, runs[0]["log"]["step"], [r["log"][metric] for r in runs], COLORS.get(name, f"C{i}"), name)
        if metric == "return" and res.get("data_return") is not None:
            _reference_lines(ax, {"dataset mean": res["data_return"]})
        ax.set_xlabel("gradient steps"); ax.set_ylabel(LABELS.get(metric, metric)); ax.grid(alpha=0.3)
        ax.set_title(f"During training (mean ± std over {len(res['seeds'])} seeds)"); ax.legend()
        _steps_axis(ax)
        if metric != "success":
            _values_axis(ax)
        if metric == "success":
            ax.set_ylim(-0.05, 1.05)
    _final_bars(bar_ax, {name: [r["final"][metric] for r in res["methods"][name]] for name in names},
                LABELS.get(metric, metric), bounded=(metric == "success"))
    if metric == "return" and res.get("data_return") is not None:
        _reference_lines(bar_ax, {"dataset mean": res["data_return"]}); bar_ax.legend(loc="lower right")
    ne = res["n_episodes"]
    bar_ax.set_title(f"Final test: mean ± std over {len(res['seeds'])} seeds\n({ne} episode{'s' if ne != 1 else ''} per seed)")
    fig.suptitle(res["dataset_id"]); plt.tight_layout()
    _finish(fig, save_path)


def plot_success_maps(maps_by_name, geom, goal, title="", save_path=None):
    """maps_by_name: {name: (n_seeds, rows, cols) success maps}; shows the average over seeds."""
    fig, axes = plt.subplots(1, len(maps_by_name), figsize=(5.5 * len(maps_by_name), 5))
    for ax, (name, maps) in zip(np.atleast_1d(axes), maps_by_name.items()):
        avg = _nanmean(np.array(maps, dtype=float), axis=0) if np.ndim(maps) == 3 else np.array(maps, dtype=float)
        draw_maze(ax, geom)
        im = ax.imshow(avg, extent=maze_extent(geom), origin="upper", cmap="RdYlGn", vmin=0, vmax=1, zorder=1)
        for (i, j), v in np.ndenumerate(avg):
            if not np.isnan(v):
                ax.text(*cell_center(geom, i, j), f"{v:.0%}", ha="center", va="center", fontsize=7, zorder=2)
        ax.plot(*goal, "*", color="gold", ms=18, mec="k", zorder=3)
        ax.set_title(f"{name}: success by start cell (mean {_nanmean(avg):.0%})")
    fig.colorbar(im, ax=axes, shrink=0.8, label="success rate")
    if title:
        fig.suptitle(title)
    _finish(fig, save_path)


def maps_from_results(res):
    return {name: [r["map"] for r in runs] for name, runs in res["methods"].items() if runs[0]["map"] is not None}


def plot_value_heatmap(task, v_params, normalizer, res=120, save_path=None):
    """Maze only: V(x, y) with the agent's body in its start pose and the goal fixed."""
    geom = task.geom
    obs, _ = task.env.reset(seed=5000)
    goal = obs["desired_goal"]
    xs = np.linspace(-geom["x_c"], geom["x_c"], res); ys = np.linspace(-geom["y_c"], geom["y_c"], res)
    X, Y = np.meshgrid(xs, ys)
    xy = np.stack([X.ravel(), Y.ravel()], -1)
    states = np.concatenate([xy, np.tile(obs["observation"], (len(xy), 1)), np.tile(goal, (len(xy), 1))], -1)
    if task.kind == "maze" and not task.is_ant:             # PointMaze: the observation starts with x, y itself
        states[:, 2:4] = xy
    V = np.array(jax.jit(value)(v_params, normalizer(states.astype(np.float32)))).reshape(res, res)
    walls = np.array(geom["walls"])
    rows = ((geom["y_c"] - Y) // geom["scale"]).astype(int).clip(0, walls.shape[0] - 1)
    cols = ((X + geom["x_c"]) // geom["scale"]).astype(int).clip(0, walls.shape[1] - 1)
    V[walls[rows, cols]] = np.nan
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_maze(ax, geom)
    im = ax.imshow(V, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower", cmap="viridis", zorder=1)
    ax.plot(*goal, "r*", ms=20, mec="white", zorder=2, label="goal")
    plt.colorbar(im, ax=ax, label="V(s)"); ax.legend(loc="upper right"); ax.set_title("IQL value function (seed 0)")
    _finish(fig, save_path)


def plot_rollouts(task, states_by_name, pool, normalizer, n=20, seed=7000, radius=0.45, save_path=None):
    """Maze only: n test episodes (the env's own start distribution) of seed 0 of each method.
    Top: the paths (black dot = start; green = reached the goal; red = failed, ending in x = flipped over
    or square = time limit). Bottom: straight-line distance to the goal over time for the same episodes."""
    k = len(states_by_name)
    fig, axes = plt.subplots(2, k, figsize=(5.2 * k, 9.2), gridspec_kw=dict(height_ratios=[5, 3]), squeeze=False)
    goal = np.asarray(task.env.reset(seed=seed)[0]["desired_goal"])
    for j, (name, st) in enumerate(states_by_name.items()):
        out = run_episodes(task, st["pi"], [(0, seed + i, None) for i in range(n)], pool, normalizer, keep_paths=True)
        ax, axd = axes[0, j], axes[1, j]
        draw_maze(ax, task.geom)
        for p, ok, fl in zip(out["paths"], out["success"], out["flipped"]):
            c = "tab:green" if ok else "tab:red"
            ax.plot(p[:, 0], p[:, 1], "-", color=c, alpha=0.55, lw=1.2, zorder=2)
            ax.plot(*p[0], "o", color="k", ms=4, zorder=3)
            if not ok:
                ax.plot(*p[-1], "x" if fl else "s", color="tab:red", ms=8 if fl else 5, mew=2, zorder=4)
            axd.plot(np.linalg.norm(p - goal, axis=1), "-", color=c, alpha=0.55, lw=1.1)
        ax.plot(*goal, "*", color="gold", ms=18, mec="k", zorder=5)
        n_fl = int(out["flipped"].sum())
        ax.set_title(f"{name} (seed 0): {out['success'].sum()}/{n} reached the goal"
                     + (f", {n_fl} flipped (x)" if task.is_ant else ""))
        axd.axhline(radius, color="gold", lw=1.5, ls="--", label=f"goal radius {radius}")
        axd.set_xlabel("time step"); axd.set_ylabel("distance to goal (straight line)")
        axd.grid(alpha=0.3); axd.legend(loc="upper right", fontsize=9)
        axd.set_title("distance to the goal over time (green = reached)")
    plt.tight_layout()
    _finish(fig, save_path)


def plot_episode_outcomes(res, save_path=None):
    """Every single episode of the final test (all seeds), per method.
    Maze: what happened in each episode (reached the goal / flipped over / time limit), and how many steps the
    successful ones needed. MuJoCo: the return of every episode (a fall shows up as a low outlier), and the
    episode lengths when some episodes end early (falls), otherwise the distribution of returns."""
    if res["kind"] != "maze":
        return _mj_plot_episode_outcomes(res, save_path)
    names = list(res["methods"])
    ep = {n: {k: np.concatenate([np.asarray(r["episodes"][k]) for r in res["methods"][n]])
              for k in res["methods"][n][0]["episodes"]} for n in names}
    T = res.get("max_steps")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))
    total = len(next(iter(ep.values()))["ret"])
    if res["kind"] == "maze":
        is_ant = "flipped" in res["metrics"]
        for i, n in enumerate(names):
            ok, fl = ep[n]["success"].astype(bool), ep[n]["flipped"].astype(bool)
            parts = [("reached the goal", ok.mean(), "tab:green")] + \
                    ([("flipped over", (fl & ~ok).mean(), "tab:red")] if is_ant else []) + \
                    [("time limit", (~ok & ~fl).mean(), "0.6")]
            bottom = 0.0
            for lab, frac, c in parts:
                a1.bar(i, frac, bottom=bottom, color=c, width=0.6, label=lab if i == 0 else None)
                if frac > 0.04:
                    a1.text(i, bottom + frac / 2, f"{frac:.0%}", ha="center", va="center", fontsize=10,
                            color="white" if c != "0.6" else "k")
                bottom += frac
            L = ep[n]["length"][ok]
            if len(L):
                a2.hist(L, bins=25, range=(0, T), histtype="step", lw=2, color=COLORS.get(n, f"C{i}"),
                        label=f"{n}: median {np.median(L):.0f} steps")
        a1.set_xticks(range(len(names)), names); a1.set_ylim(0, 1); a1.set_ylabel("fraction of test episodes")
        a1.set_title(f"Outcome of every final-test episode ({total} per method)")
        a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9, frameon=False)
        a1.set_xlim(-0.6, len(names) - 0.4)
        a2.set_xlabel("steps to reach the goal"); a2.set_ylabel("successful episodes")
        a2.set_title("How fast the successful episodes reached the goal")
        if a2.get_legend_handles_labels()[0]:
            a2.legend(fontsize=9)
        else:
            a2.text(0.5, 0.5, "no episode reached the goal", transform=a2.transAxes, ha="center", va="center",
                    fontsize=12, color="0.4")
    else:
        rng = np.random.default_rng(0)
        for i, n in enumerate(names):
            R = ep[n]["ret"]; c = COLORS.get(n, f"C{i}")
            short = ep[n]["length"] < T if T else np.zeros(len(R), bool)
            x = i + rng.uniform(-0.22, 0.22, len(R))
            a1.scatter(x[~short], R[~short], s=14, color=c, alpha=0.6, edgecolor="none")
            a1.scatter(x[short], R[short], s=30, color=c, marker="x", label="ended early (fell)" if i == 0 else None)
            a1.plot([i - 0.3, i + 0.3], [np.median(R)] * 2, color="k", lw=2)
        if res.get("data_return") is not None:
            _reference_lines(a1, {"dataset mean": res["data_return"]})
        a1.set_xticks(range(len(names)), names); a1.set_ylabel("episode return"); _values_axis(a1)
        a1.set_title(f"Return of every final-test episode ({total} per method; black = median)")
        a1.legend(fontsize=9, loc="lower right")
        any_short = T and any((ep[n]["length"] < T).any() for n in names)
        for i, n in enumerate(names):
            c = COLORS.get(n, f"C{i}")
            if any_short:
                L = ep[n]["length"]
                a2.hist(L, bins=25, range=(0, T), histtype="step", lw=2, color=c,
                        label=f"{n}: {np.mean(L < T):.0%} ended early")
            else:
                a2.hist(ep[n]["ret"], bins=25, histtype="step", lw=2, color=c, label=n)
        if any_short:
            a2.set_xlabel("episode length (steps)"); a2.set_title(f"Episode lengths ({T} = no fall)")
        else:
            a2.set_xlabel("episode return"); a2.set_title("Distribution of episode returns"); a2.xaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:,.0f}"))
        a2.set_ylabel("episodes"); a2.legend(fontsize=9)
    for ax in (a1, a2):
        ax.grid(alpha=0.3); ax.set_axisbelow(True)
    fig.suptitle(f"{res['dataset_id']}: final test, every episode")
    plt.tight_layout()
    _finish(fig, save_path)


def plot_finetune(ft_log, metrics, save_path=None):
    """Fine-tuning curves (mean +- std over seeds) for each metric; step 0 = the offline policy."""
    if _is_mujoco_metrics(metrics):
        return _mj_plot_finetune(ft_log, metrics, save_path)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for ax, m, c in zip(axes[0], metrics, ["tab:green", "tab:red", "tab:purple"]):
        _band(ax, ft_log["step"], np.array(ft_log[m], dtype=float).T, c, m)
        ax.set_xlabel("online environment steps (per seed)"); ax.set_ylabel(LABELS.get(m, m)); ax.grid(alpha=0.3)
        ax.set_title(f"{LABELS.get(m, m)} during fine-tuning (step 0 = offline)")
        _steps_axis(ax)
        if m in ("success", "flipped"):
            ax.set_ylim(-0.05, 1.05)
        else:
            _values_axis(ax)
    plt.tight_layout()
    _finish(fig, save_path)


def plot_offline_online(offline_res, ft_log, metrics, method="IQL", refs=None, save_path=None):
    """The whole learning curve of one method: offline training, then online fine-tuning continuing on the
    same x-axis (one online environment step = one gradient step), with the online part shaded.
    offline_res: the evaluate_final result of Part 1 (holds the offline training log of every seed).
    refs: optional {metric: {label: value}} reference lines, e.g. the dataset's mean return or BC's final score."""
    if _is_mujoco_metrics(metrics):
        return _mj_plot_offline_online(offline_res, ft_log, metrics, method, refs, save_path)
    runs = offline_res["methods"][method]
    off_x = np.asarray(runs[0]["log"]["step"], dtype=float)
    end = off_x[-1]
    on_x = end + np.asarray(ft_log["step"][1:], dtype=float)        # fine-tuning step 0 = the offline policy
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.5 * len(metrics), 4.3), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        off = np.array([r["log"][m] for r in runs], dtype=float)                   # (seeds, points)
        on = np.array(ft_log[m][1:], dtype=float).T                                  # (seeds, points)
        ax.axvspan(end, on_x[-1], color="tab:green", alpha=0.08, zorder=0)
        ax.axvline(end, color="k", lw=0.8, ls=":")
        _band(ax, off_x, off, COLORS["offline IQL"], f"offline {method}")
        _band(ax, np.concatenate([[end], on_x]), np.concatenate([off[:, -1:], on], 1), COLORS["fine-tuned IQL"],
              "online fine-tuning")
        _reference_lines(ax, (refs or {}).get(m))
        ax.text(end + 0.01 * (on_x[-1] - off_x[0]), 0.97, "online", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=10, color="tab:green")
        ax.set_xlabel("training steps (offline: gradient steps, then online: environment steps)")
        ax.set_ylabel(LABELS.get(m, m)); ax.grid(alpha=0.3); _steps_axis(ax)
        ax.set_title(f"{method}: {LABELS.get(m, m)}, offline then online (mean ± std over seeds)")
        if m in ("success", "flipped"):
            ax.set_ylim(-0.05, 1.05)
        else:
            _values_axis(ax)
        ax.legend(loc="lower right" if m != "flipped" else "upper right", fontsize=9)
    plt.tight_layout()
    _finish(fig, save_path)


def _run_tracked(task, pi_params, pool, normalizer, n, seed, seed_index=0):
    """MuJoCo: n deterministic test episodes of one trained seed, recording forward position, forward velocity
    and torso height at every step. Returns a list of dicts of arrays."""
    envs = pool.get(n)
    obs = [env.reset(seed=seed + i)[0] for i, env in enumerate(envs)]
    from .video import _torso_id
    tid = _torso_id(envs[0].unwrapped.model)
    rec = [dict(x=[float(env.unwrapped.data.qpos[0])], v=[0.0], z=[float(env.unwrapped.data.xpos[tid][2])], fell=False)
           for env in envs]
    active = np.ones(n, bool)
    pi = jax.tree.map(lambda p: p[seed_index:seed_index + 1], pi_params)
    while active.any():
        S = normalizer(np.stack([task.state(o) for o in obs]))
        A = np.asarray(act_all_seeds(pi, S))[0]
        for i in np.flatnonzero(active):
            obs[i], r, term, trunc, info = envs[i].step(A[i])
            d = envs[i].unwrapped.data
            rec[i]["x"].append(float(info.get("x_position", d.qpos[0])))
            rec[i]["v"].append(float(info.get("x_velocity", d.qvel[0])))
            rec[i]["z"].append(float(d.xpos[tid][2]))             # torso height above the floor (m)
            if term or trunc:
                active[i] = False; rec[i]["fell"] = bool(term)
    return [{k: (np.array(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rec]


# ================================================================================================================
# MuJoCo figures (HalfCheetah, Hopper): one consistent style. The maze figures above are unchanged.
# ================================================================================================================
from contextlib import contextmanager

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MJ_COLORS = {                    # Okabe-Ito palette: distinguishable with the common forms of color blindness
    "BC": "#0072B2", "IQL": "#E69F00", "offline IQL": "#E69F00", "fine-tuned IQL": "#009E73",
    "dataset": "#9A9A9A", "dataset mean": "#4D4D4D",
}
MJ_EXTRA = ["#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
MJ_RC = {
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.titlesize": 13, "figure.titleweight": "bold",
}
TASK_TITLES = {"halfcheetah": "HalfCheetah", "hopper": "Hopper", "walker2d": "Walker2d", "ant": "Ant"}
MJ_LABELS = {"return": "Episode return", "pct_data": "Return (% of dataset mean)", "score": "Normalized score"}


@contextmanager
def _mj_style():
    with plt.rc_context(MJ_RC):
        yield


def _mj_color(name, i=0):
    return MJ_COLORS.get(name, MJ_EXTRA[i % len(MJ_EXTRA)])


def task_title(dataset_id):
    """'mujoco/hopper/medium-v0' -> 'Hopper (medium data)'."""
    parts = dataset_id.split("/")
    env = TASK_TITLES.get(parts[1], parts[1]) if len(parts) > 2 else dataset_id
    quality = parts[2].split("-v")[0] if len(parts) > 2 else ""
    return f"{env} ({quality} data)" if quality else env


def _is_mujoco_metrics(metrics):
    return any(m in ("return", "pct_data", "score") for m in metrics)


def _mj_band(ax, x, Y, color, label, alpha=0.18):
    Y = np.asarray(Y, dtype=float)
    m, s = Y.mean(0), Y.std(0)
    lo, hi = np.maximum(m - s, Y.min(0)), np.minimum(m + s, Y.max(0))
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)
    ax.plot(x, m, "-", color=color, lw=2.0, label=label)
    ax.plot(x, m, "o", color=color, ms=3.2)


def _mj_ref_lines(ax, refs):
    styles = {"dataset mean": dict(color=MJ_COLORS["dataset mean"], ls="--", lw=1.3)}
    for label, v in (refs or {}).items():
        st = styles.get(label, dict(color=MJ_COLORS["BC"], ls=":", lw=1.6))
        ax.axhline(v, label=label, zorder=1, **st)


def _mj_bars(ax, names, means, stds, ylabel):
    tops = []
    for i, (n, m, s) in enumerate(zip(names, means, stds)):
        ax.bar(i, m, width=0.62, color=_mj_color(n, i), alpha=0.85, zorder=2, edgecolor="white", lw=0)
        ax.errorbar(i, m, yerr=s, fmt="none", ecolor="#222222", elinewidth=1.4, capsize=7, capthick=1.4, zorder=3)
        tops.append(m + s)
    lo, hi = _bar_limits(means, stds)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, tops[i] + 0.025 * (hi - lo), _fmt_pm(m, s, False), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False); _values_axis(ax)


def _bar_limits(means, stds):
    """y-limits for bars that start at 0: room for the error bars and the value labels, also for negative values."""
    top = max(0.0, max(m + s for m, s in zip(means, stds)))
    bot = min(0.0, min(m - s for m, s in zip(means, stds)))
    span = max(top - bot, 1e-9)
    return bot - (0.05 * span if bot < 0 else 0.0), top + 0.15 * span


def _bottom_legend(fig, handles, ncol=None, y=-0.02):
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles),
               frameon=False, handlelength=2.2, columnspacing=1.8)


def _mj_plot_offline_results(res, save_path=None):
    metric = res["metrics"][0]
    names = list(res["methods"])
    n_seeds, ne = len(res["seeds"]), res["n_episodes"]
    has_curves = any(runs[0]["log"] and runs[0]["log"].get(metric) for runs in res["methods"].values())
    refs = {"dataset mean": res["data_return"]} if metric == "return" and res.get("data_return") else {}
    with _mj_style():
        if has_curves:
            fig, (ax, bar_ax) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[1.7, 1]))
            for i, name in enumerate(names):
                runs = res["methods"][name]
                if runs[0]["log"] and runs[0]["log"].get(metric):
                    _mj_band(ax, runs[0]["log"]["step"], [r["log"][metric] for r in runs], _mj_color(name, i), name)
            _mj_ref_lines(ax, refs)
            ax.set_xlabel("Gradient steps"); ax.set_ylabel(MJ_LABELS.get(metric, metric)); ax.set_title("Training")
            _steps_axis(ax); _values_axis(ax)
        else:
            fig, bar_ax = plt.subplots(figsize=(6, 4.4))
        f = [np.asarray([r["final"][metric] for r in res["methods"][n]], dtype=float) for n in names]
        _mj_bars(bar_ax, names, [v.mean() for v in f], [v.std() for v in f], MJ_LABELS.get(metric, metric))
        _mj_ref_lines(bar_ax, refs)
        bar_ax.set_title(f"Final test ({ne} episodes × {n_seeds} seeds)")
        handles = [Line2D([], [], color=_mj_color(n, i), lw=2.5, label=n) for i, n in enumerate(names)] \
            if has_curves else []
        if refs:
            handles.append(Line2D([], [], color=MJ_COLORS["dataset mean"], ls="--", lw=1.3, label="Dataset mean"))
        if has_curves:
            handles.append(Patch(facecolor="0.6", alpha=0.3, label="± 1 std over seeds"))
        fig.suptitle(task_title(res["dataset_id"]))
        plt.tight_layout()
        if handles:
            _bottom_legend(fig, handles)
        _finish(fig, save_path)


def _mj_plot_episode_outcomes(res, save_path=None):
    names = list(res["methods"])
    ep = {n: {k: np.concatenate([np.asarray(r["episodes"][k]) for r in res["methods"][n]])
              for k in res["methods"][n][0]["episodes"]} for n in names}
    T = res.get("max_steps")
    for n in names:                                     # results saved before fell/stuck were recorded
        ep[n].setdefault("fell", ep[n]["length"] < T if T else np.zeros(len(ep[n]["ret"]), bool))
        ep[n].setdefault("stuck", np.zeros(len(ep[n]["ret"]), bool))
        ep[n]["fell"] = ep[n]["fell"].astype(bool); ep[n]["stuck"] = ep[n]["stuck"].astype(bool)
    total = len(next(iter(ep.values()))["ret"])
    any_fell = any(ep[n]["fell"].any() for n in names)
    rng = np.random.default_rng(0)
    with _mj_style():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
        for i, n in enumerate(names):
            R, c = ep[n]["ret"], _mj_color(n, i)
            fell, stuck = ep[n]["fell"], ep[n]["stuck"]
            ok = ~fell & ~stuck
            x = i + rng.uniform(-0.2, 0.2, len(R))
            a1.scatter(x[ok], R[ok], s=16, color=c, alpha=0.55, edgecolor="none", zorder=3)
            a1.scatter(x[fell], R[fell], s=34, color=c, marker="x", linewidths=1.6, zorder=4)
            a1.scatter(x[stuck], R[stuck], s=30, facecolor="none", edgecolor=c, marker="s", linewidths=1.4, zorder=4)
            med = np.median(R)
            a1.plot([i - 0.28, i + 0.28], [med, med], color="#222222", lw=2, zorder=5)
            a1.annotate(f"median {med:,.0f}", (i + 0.3, med), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=9, color="#222222")
        if res.get("data_return"):
            _mj_ref_lines(a1, {"dataset mean": res["data_return"]})
        a1.set_xticks(range(len(names)),
                      [f"{n}\nfell {ep[n]['fell'].mean():.0%} · stuck {ep[n]['stuck'].mean():.0%}" for n in names])
        a1.set_xlim(-0.55, len(names) - 0.2)
        a1.set_ylabel("Episode return"); _values_axis(a1); a1.grid(axis="x", visible=False)
        a1.set_title(f"Every final-test episode ({total} per method)")
        for i, n in enumerate(names):
            c = _mj_color(n, i)
            vals = ep[n]["length"] if any_fell else ep[n]["ret"]
            rng_ = (0, T) if any_fell else None
            a2.hist(vals, bins=30, range=rng_, histtype="stepfilled", color=c, alpha=0.25, zorder=2)
            a2.hist(vals, bins=30, range=rng_, histtype="step", color=c, lw=1.8, zorder=3)
        if any_fell:
            a2.set_xlabel(f"Episode length (steps; {T} = no fall)"); a2.set_title("Episode lengths")
        else:
            a2.set_xlabel("Episode return"); a2.set_title("Distribution of episode returns")
            a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        a2.set_ylabel("Episodes")
        handles = [Patch(facecolor=_mj_color(n, i), alpha=0.85, label=n) for i, n in enumerate(names)]
        handles += [Line2D([], [], ls="none", marker="o", color="0.4", ms=5, label="Completed"),
                    Line2D([], [], ls="none", marker="x", color="0.2", ms=7, mew=1.6, label="Fell (episode ended)"),
                    Line2D([], [], ls="none", marker="s", mfc="none", mec="0.2", ms=6, label="Stuck (stopped moving)"),
                    Line2D([], [], color="#222222", lw=2, label="Median")]
        if res.get("data_return"):
            handles.append(Line2D([], [], color=MJ_COLORS["dataset mean"], ls="--", lw=1.3, label="Dataset mean"))
        fig.suptitle(f"{task_title(res['dataset_id'])}: final test, episode by episode")
        plt.tight_layout()
        _bottom_legend(fig, handles, ncol=min(len(handles), 4))
        _finish(fig, save_path)


def _mj_plot_finetune(ft_log, metrics, save_path=None):
    m = metrics[0]
    with _mj_style():
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        _mj_band(ax, ft_log["step"], np.array(ft_log[m], dtype=float).T, MJ_COLORS["fine-tuned IQL"], "fine-tuned IQL")
        ax.set_xlabel("Online environment steps (per seed)"); ax.set_ylabel(MJ_LABELS.get(m, m))
        ax.set_title("Online fine-tuning (step 0 = offline policy; mean ± std over seeds)")
        _steps_axis(ax); _values_axis(ax)
        plt.tight_layout()
        _finish(fig, save_path)


def _mj_plot_offline_online(offline_res, ft_log, metrics, method="IQL", refs=None, save_path=None):
    m = metrics[0]
    runs = offline_res["methods"][method]
    off_x = np.asarray(runs[0]["log"]["step"], dtype=float)
    end = off_x[-1]
    on_x = end + np.asarray(ft_log["step"][1:], dtype=float)
    off = np.array([r["log"][m] for r in runs], dtype=float)
    on = np.array(ft_log[m][1:], dtype=float).T
    with _mj_style():
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.axvspan(end, on_x[-1], color=MJ_COLORS["fine-tuned IQL"], alpha=0.07, lw=0, zorder=0)
        ax.axvline(end, color="0.3", lw=0.9, ls=":", zorder=1)
        _mj_band(ax, off_x, off, MJ_COLORS["offline IQL"], f"{method}, offline")
        _mj_band(ax, np.concatenate([[end], on_x]), np.concatenate([off[:, -1:], on], 1),
                 MJ_COLORS["fine-tuned IQL"], f"{method}, online fine-tuning")
        r = dict((refs or {}).get(m) or {})
        _mj_ref_lines(ax, r)
        ax.text(0.5 * (off_x[0] + end), 1.01, "offline", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=10, color="0.35")
        ax.text(0.5 * (end + on_x[-1]), 1.01, "online", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=10, color=MJ_COLORS["fine-tuned IQL"])
        ax.set_xlabel("Training steps (offline: gradient steps; online: environment steps)")
        ax.set_ylabel(MJ_LABELS.get(m, m)); _steps_axis(ax); _values_axis(ax)
        handles = [Line2D([], [], color=MJ_COLORS["offline IQL"], lw=2.5, label=f"{method}, offline"),
                   Line2D([], [], color=MJ_COLORS["fine-tuned IQL"], lw=2.5, label=f"{method}, online fine-tuning")]
        for label in r:
            st = dict(color=MJ_COLORS["dataset mean"], ls="--", lw=1.3) if label == "dataset mean" else \
                dict(color=MJ_COLORS["BC"], ls=":", lw=1.6)
            handles.append(Line2D([], [], label=label[0].upper() + label[1:], **st))
        handles.append(Patch(facecolor="0.6", alpha=0.3, label="± 1 std over seeds"))
        fig.suptitle(f"{task_title(offline_res['dataset_id'])}: offline training, then online fine-tuning", y=1.04)
        plt.tight_layout()
        _bottom_legend(fig, handles, ncol=min(len(handles), 3))
        _finish(fig, save_path)


def plot_mujoco_trajectories(task, states_by_name, pool, normalizer, n=5, seed=7000, data_episodes=5,
                             smooth_s=0.5, save_path=None):
    """MuJoCo: n test episodes per method (trained seed 0) next to dataset episodes (gray): forward position,
    forward velocity (moving average over `smooth_s` seconds) and torso height, against time in seconds.
    x marks an episode that ended with a fall, a square one that got stuck (stopped moving)."""
    from .evaluation import STUCK_SPEED, STUCK_WINDOW
    u = task.env.unwrapped
    nq, dt = u.model.nq, u.dt
    from .video import _torso_id
    tid = _torso_id(u.model)
    u_env = task.env
    u_env.reset(seed=seed)
    z_offset = float(u.data.xpos[tid][2] - u.data.qpos[1])     # observation[0] = qpos[1]; plot the torso's height
    w = max(1, int(round(smooth_s / dt)))
    kern = np.ones(w)

    def sm(v):                                             # centered moving average, same length, edge-corrected
        v = np.asarray(v, dtype=float)
        k = kern[:max(1, min(len(kern), len(v)))]          # shorter window for episodes shorter than the window
        return np.convolve(v, k, mode="same") / np.convolve(np.ones_like(v), k, mode="same")
    rng = np.random.default_rng(seed)
    eps = list(task.dataset.iterate_episodes(rng.choice(task.dataset.total_episodes, data_episodes,
                                                         replace=False).tolist()))
    with _mj_style():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        for ep in eps:                                     # observation = qpos[1:] followed by qvel
            o = np.asarray(ep.observations)
            v, z = o[:, nq - 1], o[:, 0] + z_offset
            x = np.concatenate([[0.0], np.cumsum(v[1:]) * dt])
            t = np.arange(len(x)) * dt
            axes[0].plot(t, x, color=MJ_COLORS["dataset"], lw=1.1, alpha=0.8)
            axes[1].plot(t, sm(v), color=MJ_COLORS["dataset"], lw=1.1, alpha=0.8)
            axes[2].plot(t, z, color=MJ_COLORS["dataset"], lw=1.0, alpha=0.7)
        n_fell, n_stuck = {}, {}
        for j, (name, st) in enumerate(states_by_name.items()):
            c = _mj_color(name, j)
            recs = _run_tracked(task, st["pi"], pool, normalizer, n, seed)
            n_fell[name] = n_stuck[name] = 0
            for r in recs:
                t = np.arange(len(r["x"])) * dt
                r_z = r["z"]
                ww = min(STUCK_WINDOW, len(r["x"]) - 1)
                stuck = (not r["fell"]) and ww > 0 and (r["x"][-1] - r["x"][-1 - ww]) / (ww * dt) < STUCK_SPEED
                n_fell[name] += r["fell"]; n_stuck[name] += stuck
                axes[0].plot(t, r["x"], color=c, lw=1.4, alpha=0.85)
                vt = t
                axes[1].plot(vt, sm(r["v"]), color=c, lw=1.4, alpha=0.85)
                axes[2].plot(t, r_z, color=c, lw=1.1, alpha=0.75)
                if r["fell"] or stuck:
                    mk = dict(marker="x", ms=8, mew=1.8, color=c) if r["fell"] else \
                        dict(marker="s", ms=7, mew=1.5, mfc="none", mec=c)
                    axes[0].plot(t[-1], r["x"][-1], ls="none", **mk)
                    axes[1].plot(vt[-1], sm(r["v"])[-1], ls="none", **mk)
                    axes[2].plot(t[-1], r_z[-1], ls="none", **mk)
        for ax, (title, yl) in zip(axes, [("Forward position", "x (m)"),
                                          (f"Forward velocity ({smooth_s:g} s moving average)", "v (m/s)"),
                                          ("Torso height", "z (m)")]):
            ax.set_xlabel("Time (s)"); ax.set_ylabel(yl); ax.set_title(title)
        handles = [Line2D([], [], color=MJ_COLORS["dataset"], lw=2, label=f"Dataset ({data_episodes} episodes)")]
        handles += [Line2D([], [], color=_mj_color(nm, j), lw=2.5,
                           label=f"{nm} ({n} episodes; fell {n_fell[nm]}, stuck {n_stuck[nm]})")
                    for j, nm in enumerate(states_by_name)]
        handles += [Line2D([], [], ls="none", marker="x", color="0.2", ms=7, mew=1.6, label="Fell"),
                    Line2D([], [], ls="none", marker="s", mfc="none", mec="0.2", ms=6, label="Stuck")]
        fig.suptitle(f"{task_title(task.cfg['dataset_id'])}: sample trajectories (trained seed 0)")
        plt.tight_layout()
        _bottom_legend(fig, handles, ncol=min(len(handles), 3))
        _finish(fig, save_path)


def plot_ablation(results, param, default=None, metric="return", save_path=None):
    """Hyperparameter study. results: {task_name: {value: evaluate_final result (IQL only)}} for one parameter
    (e.g. 'discount' or 'beta'). One panel per task: final test mean ± std over seeds for each value; the default
    value is outlined."""
    sym = {"discount": "γ (discount)", "beta": "β (inverse temperature)", "expectile": "τ (expectile)"}.get(param, param)
    tasks = list(results)
    with _mj_style():
        fig, axes = plt.subplots(1, len(tasks), figsize=(5.6 * len(tasks), 4.2), squeeze=False)
        for ax, tname in zip(axes[0], tasks):
            vals = sorted(results[tname])
            f = [np.asarray([r["final"][metric] for r in results[tname][v]["methods"]["IQL"]], dtype=float)
                 for v in vals]
            means, stds = [x.mean() for x in f], [x.std() for x in f]
            for i, (v, mu, sd) in enumerate(zip(vals, means, stds)):
                is_def = default is not None and np.isclose(v, default)
                ax.bar(i, mu, width=0.62, color=MJ_COLORS["IQL"], alpha=0.85 if is_def else 0.5, zorder=2,
                       edgecolor="#222222" if is_def else "none", lw=1.4 if is_def else 0)
                ax.errorbar(i, mu, yerr=sd, fmt="none", ecolor="#222222", elinewidth=1.4, capsize=7, capthick=1.4,
                            zorder=3)
                lo, hi = _bar_limits(means, stds)
                ax.text(i, max(mu + sd, 0.0) + 0.025 * (hi - lo), _fmt_pm(mu, sd, False),
                        ha="center", va="bottom", fontsize=10)
            ax.set_xticks(range(len(vals)), [f"{v:g}" + (" (default)" if default is not None and np.isclose(v, default)
                                                        else "") for v in vals])
            ax.set_xlabel(sym); ax.set_ylabel(MJ_LABELS.get(metric, metric)); ax.set_title(tname)
            ax.set_ylim(*_bar_limits(means, stds)); _values_axis(ax)
            ax.grid(axis="x", visible=False)
        fig.suptitle(f"IQL, offline: effect of {sym} (final test, mean ± std over seeds)", y=1.03)
        plt.tight_layout()
        _finish(fig, save_path)


def plot_dataset_returns(task, save_path=None):
    """MuJoCo: distribution of the dataset's episode returns, with the mean and the fraction of episodes that end
    with a termination (fall)."""
    eps = list(task.dataset.iterate_episodes())
    rets = np.array([float(np.sum(ep.rewards)) for ep in eps])
    fell = np.array([bool(np.any(ep.terminations)) for ep in eps])
    with _mj_style():
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.hist(rets, bins=40, histtype="stepfilled", color=MJ_COLORS["dataset"], alpha=0.45)
        ax.hist(rets, bins=40, histtype="step", color="#555555", lw=1.2)
        ax.axvline(rets.mean(), color=MJ_COLORS["dataset mean"], ls="--", lw=1.3)
        ax.annotate(f"mean {rets.mean():,.0f}", (rets.mean(), 0.97), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", va="top", fontsize=10, color="#333333")
        ax.set_xlabel("Episode return"); ax.set_ylabel("Episodes"); ax.grid(axis="x", visible=False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_title(f"{task_title(task.cfg['dataset_id'])}: dataset, {len(rets):,} episodes"
                     + (f", {fell.mean():.0%} end with a fall" if fell.any() else ""))
        plt.tight_layout()
        _finish(fig, save_path)


def plot_ablation_curves(results, param, metric="return", save_path=None):
    """Hyperparameter study: training curves (mean ± std over seeds) for each value of `param`, one panel per task.
    results: {task_name: {value: evaluate_final result (IQL only)}}."""
    sym = {"discount": "γ", "beta": "β", "expectile": "τ"}.get(param, param)
    tasks = list(results)
    shades = ["#56B4E9", "#E69F00", "#7A1F5C"]               # small, default (orange, as IQL elsewhere), large
    with _mj_style():
        fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.2), squeeze=False)
        handles = {}
        for ax, tname in zip(axes[0], tasks):
            vals = sorted(results[tname])
            for k, v in enumerate(vals):
                runs = results[tname][v]["methods"]["IQL"]
                c = shades[k] if len(vals) == 3 else MJ_EXTRA[k % len(MJ_EXTRA)]
                _mj_band(ax, runs[0]["log"]["step"], [r["log"][metric] for r in runs], c, f"{sym} = {v:g}",
                         alpha=0.22)
                handles[v] = Line2D([], [], color=c, lw=2.5, label=f"{sym} = {v:g}")
            ax.set_xlabel("Gradient steps"); ax.set_ylabel(MJ_LABELS.get(metric, metric)); ax.set_title(tname)
            _steps_axis(ax); _values_axis(ax)
        fig.suptitle(f"IQL, offline: training curves for different {sym} (mean ± std over seeds)", y=1.03)
        plt.tight_layout()
        _bottom_legend(fig, [handles[v] for v in sorted(handles)])
        _finish(fig, save_path)
