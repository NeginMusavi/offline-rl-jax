"""Hyperparameter study for offline IQL: one factor at a time around the default configuration.

Each setting is trained for all seeds in parallel and evaluated with the same protocol as the main notebooks.
Results are written to one JSON file per setting as soon as the setting is finished; a rerun skips settings whose
file exists, so an interrupted sweep resumes where it stopped.
"""
import os

import numpy as np

from .checkpoint import load_json, save_json
from .config import load_config
from .data import Normalizer, to_device
from .evaluation import EnvPool, evaluate, evaluate_final
from .tasks import Task
from .training import train_offline


def one_at_a_time(values, defaults):
    """values: {param: [v1, v2, ...]}, defaults: {param: v}. Returns the list of settings (dicts) that vary one
    parameter at a time with the others at their defaults; the default setting appears once."""
    settings = [dict(defaults)]
    for p, vs in values.items():
        for v in vs:
            s = {**defaults, p: v}
            if s not in settings:
                settings.append(s)
    return settings


def setting_key(setting):
    return "_".join(f"{k}{v:g}" for k, v in sorted(setting.items()))


def run_sweep(config_file, settings, out_dir, overrides=None, verbose=True):
    """Trains and evaluates offline IQL for every setting. Returns {setting_key: evaluate_final result}.
    overrides: config values applied to every setting (e.g. iql_steps)."""
    cfg0, _ = load_config(config_file, **(overrides or {}))
    os.makedirs(out_dir, exist_ok=True)
    todo = [s for s in settings if not os.path.exists(f"{out_dir}/{setting_key(s)}.json")]
    results = {}
    if todo:
        task = Task(cfg0, verbose=verbose)
        raw = task.transitions()
        norm = Normalizer(raw["states"])
        data, pool, n = to_device(raw, norm), EnvPool(task), len(cfg0["seeds"])
    for i, s in enumerate(settings):
        key, path = setting_key(s), f"{out_dir}/{setting_key(s)}.json"
        if os.path.exists(path):
            results[key] = load_json(path)
            if verbose:
                print(f"[{i + 1}/{len(settings)}] {key}: loaded {path}")
            continue
        cfg = {**cfg0, **s}
        if verbose:
            print(f"[{i + 1}/{len(settings)}] {key}: training IQL for {cfg['iql_steps']:,} steps")
        ev = lambda pi: evaluate(task, pi, n, cfg["eval_episodes"], 1000, pool, norm)
        states, log = train_offline("iql", cfg, data, ev, verbose=verbose)
        res = evaluate_final(task, cfg, {"IQL": states}, {"IQL": log}, pool, norm, maps=False)
        res["setting"] = s
        save_json(path, res)
        results[key] = res
    return results


def summary_rows(results, metric="return"):
    """One row per setting: mean and std over seeds of the final metric, and the fractions of episodes that fell
    or got stuck."""
    rows = []
    for key, res in results.items():
        runs = res["methods"]["IQL"]
        f = np.array([r["final"][metric] for r in runs], dtype=float)
        eps = lambda k: np.mean([np.mean(r["episodes"][k]) for r in runs]) if k in runs[0]["episodes"] else np.nan
        rows.append(dict(key=key, **res["setting"], mean=float(f.mean()), std=float(f.std()),
                         pct_data=float(np.mean([r["final"].get("pct_data", np.nan) for r in runs])),
                         fell=float(eps("fell")), stuck=float(eps("stuck"))))
    return rows


def print_summary(task_name, results, metric="return"):
    rows = summary_rows(results, metric)
    params = [k for k in rows[0] if k not in ("key", "mean", "std", "pct_data", "fell", "stuck")]
    print(task_name)
    print("".join(f"{p:>10s}" for p in params) + f"{metric:>22s}{'% data':>10s}{'fell':>8s}{'stuck':>8s}")
    for r in rows:
        print("".join(f"{r[p]:>10g}" for p in params) + f"{r['mean']:>14,.0f} ± {r['std']:<5,.0f}"
              + f"{r['pct_data']:>10.1f}{r['fell']:>8.0%}{r['stuck']:>8.0%}")
    return rows


def by_value(results, param, defaults):
    """{value of `param`: result} for the settings where only `param` differs from the defaults."""
    out = {}
    for res in results.values():
        s = res["setting"]
        if all(np.isclose(s[k], v) for k, v in defaults.items() if k != param):
            out[s[param]] = res
    return out
