"""Saving and loading trained networks (pickle of numpy arrays) and results (JSON)."""
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def save_checkpoint(path, **items):
    """Items whose name ends in "states" are network/optimizer pytrees and are moved to the CPU first."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    items = {k: (jax.device_get(v) if k.endswith("states") else v) for k, v in items.items()}
    with open(path, "wb") as f:
        pickle.dump(items, f)
    print(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def load_checkpoint(path):
    with open(path, "rb") as f:
        items = pickle.load(f)
    for k in items:
        if k.endswith("states"):
            items[k] = jax.tree.map(jnp.asarray, items[k])
    print("loaded", path, "| contents:", ", ".join(items))
    return items


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return np.asarray(x).tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=1)
    print("saved", path)


def load_json(path):
    with open(path) as f:
        return json.load(f)
