"""Command-line version of the notebooks.

    python scripts/run.py --config configs/hopper_medium.yaml                 # offline, then online
    python scripts/run.py --config configs/hopper_medium.yaml --stage offline
    python scripts/run.py --config configs/hopper_medium.yaml --stage online  # loads checkpoints/<name>/offline.pkl
    python scripts/run.py --config configs/antmaze_medium_play.yaml --set seeds=[0] iql_steps=500000

--set overrides config values (YAML syntax). Keys that exist in both sections (eval_every, eval_episodes,
final_episodes) set the offline value; write online.<key>=... for the online one.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")

from offrl import (EnvPool, Normalizer, Task, evaluate, evaluate_final, load_checkpoint, load_config,  # noqa: E402
                   save_checkpoint, save_json, select_seed, summarize, to_device, train_offline)
from offrl import plotting  # noqa: E402
from offrl.algos.finetune import finetune  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--stage", choices=["offline", "online", "all"], default="all")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg, online = load_config(args.config, **overrides)
    res_dir, ckpt_dir = os.path.join(args.results_dir, cfg["name"]), os.path.join(args.ckpt_dir, cfg["name"])
    ck = None
    if args.stage == "online":
        ck = load_checkpoint(os.path.join(ckpt_dir, "offline.pkl"))
        cfg = ck["config"]

    task = Task(cfg)
    raw = task.transitions()
    norm = Normalizer(raw["states"]) if ck is None else Normalizer.from_state_dict(ck["normalizer"])
    data, pool, n = to_device(raw, norm), EnvPool(task), len(cfg["seeds"])

    if args.stage in ("offline", "all"):
        ev = lambda pi: evaluate(task, pi, n, cfg["eval_episodes"], 1000, pool, norm)
        bc, bc_log = train_offline("bc", cfg, data, ev)
        iql, iql_log = train_offline("iql", cfg, data, ev)
        res = evaluate_final(task, cfg, {"BC": bc, "IQL": iql}, {"BC": bc_log, "IQL": iql_log}, pool, norm)
        summarize(res)
        save_json(f"{res_dir}/offline_results.json", res)
        plotting.plot_offline_results(res, save_path=f"{res_dir}/figures/offline.png")
        if task.kind == "maze":
            plotting.plot_success_maps(plotting.maps_from_results(res), task.geom, res["goal"],
                                       save_path=f"{res_dir}/figures/offline_success_maps.png")
            plotting.plot_value_heatmap(task, select_seed(iql["v"], 0), norm, save_path=f"{res_dir}/figures/iql_value.png")
        save_checkpoint(f"{ckpt_dir}/offline.pkl", config=cfg, online=online, normalizer=norm.state_dict(),
                        bc_states=bc, iql_states=iql)
    else:
        iql = ck["iql_states"]

    if args.stage in ("online", "all") and not online:
        print(f"{args.config} has no `online` section: skipping fine-tuning.")
    elif args.stage in ("online", "all"):
        def save_progress(states, log):
            save_checkpoint(f"{ckpt_dir}/finetuned.pkl", config=cfg, online=online, normalizer=norm.state_dict(),
                            iql_states=states, log=log)
        ft, ft_log = finetune(task, cfg, online, iql, data, pool, norm, on_eval=save_progress)
        save_json(f"{res_dir}/finetune_log.json", ft_log)
        plotting.plot_finetune(ft_log, list(task.curve_metrics), save_path=f"{res_dir}/figures/finetune_curves.png")
        res = evaluate_final(task, {**cfg, "final_episodes": online["final_episodes"]},
                             {"offline IQL": iql, "fine-tuned IQL": ft}, None, pool, norm)
        summarize(res)
        save_json(f"{res_dir}/finetune_results.json", res)


if __name__ == "__main__":
    main()
