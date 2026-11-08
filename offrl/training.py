"""Fast training loops: jax.lax.scan runs many updates inside one compiled program, and
jax.vmap trains several seeds at once (every state has a leading "seed" axis)."""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from .algos.bc import make_bc
from .algos.iql import make_iql
from .evaluation import format_metrics

MAKERS = {"bc": make_bc, "iql": make_iql}


def seed_keys(seeds, salt):
    """One PRNG key per seed; `salt` separates different uses (init, training, ...)."""
    return jnp.stack([jax.random.fold_in(jax.random.PRNGKey(s), salt) for s in seeds])


def split_keys(keys):
    both = jax.vmap(jax.random.split)(keys)
    return both[:, 0], both[:, 1]


def select_seed(tree, i):
    """Take seed i out of a stacked pytree."""
    return jax.tree.map(lambda x: x[i], tree)


def make_trainer(update, batch_size):
    """train_chunk(states, data, keys, n_steps): n_steps updates per seed, sampling from `data`."""
    def one_step(state, data, key):
        idx = jax.random.randint(key, (batch_size,), 0, data["s"].shape[0])
        return update(state, jax.tree.map(lambda x: x[idx], data))

    def one_seed(state, data, key, n_steps):
        def body(st, k):
            return one_step(st, data, k)
        state, metrics = jax.lax.scan(body, state, jax.random.split(key, n_steps))
        return state, jax.tree.map(jnp.mean, metrics)

    @functools.partial(jax.jit, static_argnums=3)
    def train_chunk(states, data, keys, n_steps):
        return jax.vmap(one_seed, in_axes=(0, None, 0, None))(states, data, keys, n_steps)
    return train_chunk


def sample_mixed_batch(off, on, size, key, batch_size):
    """Uniform batch over the offline data followed by the first `size` online transitions."""
    n_off = off["s"].shape[0]
    idx = jax.random.randint(key, (batch_size,), 0, n_off + size)
    from_off = idx < n_off
    i_off = jnp.minimum(idx, n_off - 1)
    i_on = jnp.clip(idx - n_off, 0, on["s"].shape[0] - 1)

    def pick(o, n):
        sel = from_off.reshape((-1,) + (1,) * (o.ndim - 1))
        return jnp.where(sel, o[i_off], n[i_on])
    return {k: pick(off[k], on[k]) for k in off}


def make_mixed_trainer(update, batch_size):
    """train_chunk(states, off, on, size, keys, n_steps): like make_trainer, but each batch is drawn
    uniformly from the shared offline data plus this seed's online buffer (first `size` entries)."""
    def one_seed(state, off, on, size, key, n_steps):
        def body(st, k):
            return update(st, sample_mixed_batch(off, on, size, k, batch_size))
        state, metrics = jax.lax.scan(body, state, jax.random.split(key, n_steps))
        return state, jax.tree.map(jnp.mean, metrics)

    @functools.partial(jax.jit, static_argnums=5)
    def train_chunk(states, off, on, size, keys, n_steps):
        return jax.vmap(one_seed, in_axes=(0, None, 0, None, 0, None))(states, off, on, size, keys, n_steps)
    return train_chunk


def train_offline(kind, cfg, data, evaluate_fn=None, verbose=True):
    """Trains BC or IQL for every seed in cfg["seeds"] at once.

    data: dict of GPU arrays s, a, r, s2, d (normalized states).
    evaluate_fn(pi_params) -> dict of per-seed metric arrays, called every cfg["eval_every"] steps.
    Returns (states, log) where log = {"step": [...], <metric>: [[per seed], ...], "train": [...]}.
    """
    seeds = cfg["seeds"]
    init, update, _ = MAKERS[kind](cfg, data["s"].shape[1], data["a"].shape[1])
    train_chunk = make_trainer(update, cfg["batch_size"])
    states = jax.vmap(init)(seed_keys(seeds, 0))
    keys = seed_keys(seeds, 1)
    total, done = cfg[f"{kind}_steps"], 0
    log = {"step": [], "train": []}
    t0, t_train = time.time(), 0.0
    while done < total:
        n = min(cfg["eval_every"], total - done)
        keys, sub = split_keys(keys)
        t1 = time.time()
        states, metrics = train_chunk(states, data, sub, n)
        jax.block_until_ready(states)
        t_train += time.time() - t1
        done += n
        m = jax.tree.map(lambda x: np.asarray(x).tolist(), metrics)
        log["step"].append(done); log["train"].append(m)
        ev = evaluate_fn(states["pi"]) if evaluate_fn is not None else {}
        for k, v in ev.items():
            log.setdefault(k, []).append(np.asarray(v).tolist())
        if verbose:
            loss = f"loss {np.mean(m['loss']):.4f}" if kind == "bc" else \
                   f"q_loss {np.mean(m['q_loss']):.3f} | V {np.mean(m['v_mean']):8.2f}"
            evs = format_metrics(ev)
            print(f"[{kind.upper():3s}] step {done:>9,} | {loss} | {evs} | train {t_train:.0f}s, "
                  f"total {time.time() - t0:.0f}s")
    return states, log
