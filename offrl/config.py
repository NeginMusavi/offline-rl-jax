"""Experiment configs (configs/*.yaml).

A config file has four parts:
    name, dataset_id, seeds   what to run
    task:                     how the data is labeled (task-specific)
    offline:                  BC/IQL hyperparameters and offline evaluation
    online:                   fine-tuning settings
load_config returns (cfg, online): cfg is one flat dict (name, dataset_id, seeds, task and offline keys),
which is what every function in this package takes.
"""
import copy

import yaml


def load_config(path, **overrides):
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = {"name": raw["name"], "dataset_id": raw["dataset_id"], "seeds": list(raw["seeds"])}
    cfg.update(raw.get("task") or {})
    cfg.update(raw.get("offline") or {})
    online = copy.deepcopy(raw.get("online") or {})
    for k, v in overrides.items():           # "online.<key>" -> online settings; other keys -> cfg unless only online has them
        if k.startswith("online."):
            online[k[len("online."):]] = v
        elif k in cfg or k not in online:
            cfg[k] = v
        else:
            online[k] = v
    return cfg, online
