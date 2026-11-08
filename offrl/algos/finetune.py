"""Offline-to-online fine-tuning of IQL, following the official IQL code:

- start from the offline networks with fresh optimizers and a constant learning rate
  (the offline policy's cosine schedule ends at 0, which would freeze it);
- collect with several environments per seed, sampling actions from the policy's Gaussian
  (`explore_temperature` scales its std);
- label new transitions exactly like the offline data (Task.online_label);
- one replay buffer per seed = all offline data + the online transitions, sampled uniformly;
- `updates_per_step` gradient updates per collected transition, with the same IQL update.

AntMaze: with reset_on_flip, an exploration episode ends when the ant flips over. That reset is
stored as a time-limit cut, not a terminal: a terminal without a penalty would make flipping look
like a cheap way to end the episode.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..evaluation import evaluate, format_metrics
from ..networks import policy_mean, policy_std
from ..training import make_mixed_trainer, seed_keys, split_keys
from .iql import make_iql


@jax.jit
def sample_all_seeds(pi, S, keys, temperature=1.0):
    """pi stacked over seeds; S: (n_seeds, n_envs, s_dim) normalized states; one key per seed."""
    def one(p, s, k):
        mean = policy_mean(p, s)
        return jnp.clip(mean + temperature * policy_std(p) * jax.random.normal(k, mean.shape), -1, 1)
    return jax.vmap(one)(pi, S, keys)


def empty_online_buffer(n_seeds, capacity, s_dim, a_dim):
    z = lambda *shape: jnp.zeros((n_seeds, capacity) + shape, jnp.float32)
    return dict(s=z(s_dim), s2=z(s_dim), a=z(a_dim), r=z(), d=z())


@jax.jit
def buffer_insert(buf, new, pos):
    """Writes `new` (n_seeds, n_new, ...) into the GPU buffer at position `pos` (no round trip)."""
    return {k: jax.lax.dynamic_update_slice_in_dim(buf[k], new[k], pos, axis=1) for k in buf}


class OnlineCollector:
    """envs_per_seed environments for each seed, stepped together."""
    def __init__(self, task, n_seeds, envs_per_seed, reset_on_flip=True, first_seed=100_000):
        self.task, self.n_seeds, self.k, self.reset_on_flip = task, n_seeds, envs_per_seed, reset_on_flip
        self.envs = [[task.make_env() for _ in range(envs_per_seed)] for _ in range(n_seeds)]
        self.next_seed = first_seed
        self.obs = [[self._reset(e) for e in row] for row in self.envs]
        self.ep_ret = np.zeros((n_seeds, envs_per_seed))
        self.stats = self._new_stats()

    def _new_stats(self):
        return [dict(episodes=0, success=0, flips=0, returns=[]) for _ in range(self.n_seeds)]

    def _reset(self, env):
        self.next_seed += 1
        return env.reset(seed=self.next_seed)[0]

    def pop_stats(self):
        out, self.stats = self.stats, self._new_stats()
        return out

    def collect(self, pi, n_vec_steps, key, normalizer, temperature=1.0):
        n, k, task = self.n_seeds, self.k, self.task
        s_dim = task.state(self.obs[0][0]).shape[0]
        a_dim = self.envs[0][0].action_space.shape[0]
        S = np.zeros((n, n_vec_steps, k, s_dim), np.float32); S2 = np.zeros_like(S)
        A = np.zeros((n, n_vec_steps, k, a_dim), np.float32)
        R = np.zeros((n, n_vec_steps, k), np.float32); D = np.zeros_like(R)
        for t in range(n_vec_steps):
            raw = np.stack([[task.state(o) for o in row] for row in self.obs])
            key, sub = jax.random.split(key)
            acts = np.asarray(sample_all_seeds(pi, normalizer(raw), jax.random.split(sub, n), temperature))
            for i in range(n):
                for j in range(k):
                    env = self.envs[i][j]
                    o2, r, term, trunc, info = env.step(acts[i, j])
                    reward, done = task.online_label(r, term, info)
                    S[i, t, j], A[i, t, j], S2[i, t, j] = raw[i, j], acts[i, j], task.state(o2)
                    R[i, t, j], D[i, t, j] = reward, float(done)
                    self.ep_ret[i, j] += r
                    if task.episode_over(r, term, trunc, info, o2, self.reset_on_flip):
                        st = self.stats[i]
                        st["episodes"] += 1; st["success"] += int(task.success(r, info))
                        st["flips"] += int(task.flipped(o2)); st["returns"].append(float(self.ep_ret[i, j]))
                        self.ep_ret[i, j] = 0.0
                        self.obs[i][j] = self._reset(env)
                    else:
                        self.obs[i][j] = o2
        flat = lambda x: x.reshape((n, n_vec_steps * k) + x.shape[3:])
        new = dict(s=normalizer(flat(S)), a=flat(A), r=flat(R), s2=normalizer(flat(S2)), d=flat(D))
        return jax.tree.map(jnp.asarray, new), key


def finetune(task, cfg, ocfg, start_states, data, pool, normalizer, on_eval=None, verbose=True):
    """Fine-tunes IQL online for every seed at once.

    cfg: the offline config; ocfg: online settings (online_steps, envs_per_seed, chunk, updates_per_step,
    reset_on_flip, explore_temperature, eval_every, eval_episodes).
    on_eval(states, log) is called after every evaluation (e.g. to save a checkpoint).
    Returns (states, log); log["step"][0] = 0 is the offline policy.
    """
    assert ocfg["chunk"] % ocfg["envs_per_seed"] == 0, "chunk must be a multiple of envs_per_seed"
    assert ocfg["online_steps"] % ocfg["chunk"] == 0 and ocfg["eval_every"] % ocfg["chunk"] == 0, \
        "online_steps and eval_every must be multiples of chunk"
    n_seeds = len(cfg["seeds"]); s_dim, a_dim = data["s"].shape[1], data["a"].shape[1]
    _, update, fresh_opts = make_iql({**cfg, "actor_lr_schedule": "constant"}, s_dim, a_dim)
    states = jax.vmap(fresh_opts)(start_states)
    trainer = make_mixed_trainer(update, cfg["batch_size"])
    buf = empty_online_buffer(n_seeds, ocfg["online_steps"], s_dim, a_dim)
    collector = OnlineCollector(task, n_seeds, ocfg["envs_per_seed"], ocfg.get("reset_on_flip", True))
    key, keys = jax.random.PRNGKey(2024), seed_keys(cfg["seeds"], 3)
    log = {"step": [], "explore": [], "train": []}
    t0 = time.time()

    def evaluate_now(step, metrics=None):
        ev = evaluate(task, states["pi"], n_seeds, ocfg["eval_episodes"], 1000, pool, normalizer)
        ex = collector.pop_stats()
        log["step"].append(step); log["explore"].append(ex)
        log["train"].append(jax.tree.map(lambda x: np.asarray(x).tolist(), metrics) if metrics is not None else None)
        for k, v in ev.items():
            log.setdefault(k, []).append(np.asarray(v).tolist())
        if verbose:
            evs = format_metrics(ev)
            n_ep = sum(e["episodes"] for e in ex)
            if n_ep and task.kind == "maze":
                ex_txt = f"explore: {n_ep} episodes, goal {sum(e['success'] for e in ex) / n_ep:.2f}" + \
                         (f", flips {sum(e['flips'] for e in ex) / n_ep:.2f}" if task.is_ant else "")
            elif n_ep:
                ex_txt = f"explore: {n_ep} episodes, return {np.mean([r for e in ex for r in e['returns']]):.1f}"
            else:
                ex_txt = "explore: -"
            loss = f" | q_loss {np.mean(metrics['q_loss']):.3f} V {np.mean(metrics['v_mean']):8.2f}" if metrics is not None else ""
            print(f"[FT] online step {step:>9,} | {evs} | {ex_txt}{loss} | {time.time() - t0:.0f}s")
        if on_eval is not None and step > 0:
            on_eval(states, log)

    evaluate_now(0)
    for start in range(0, ocfg["online_steps"], ocfg["chunk"]):
        new, key = collector.collect(states["pi"], ocfg["chunk"] // ocfg["envs_per_seed"], key, normalizer,
                                     ocfg.get("explore_temperature", 1.0))
        buf = buffer_insert(buf, new, start)
        keys, sub = split_keys(keys)
        step = start + ocfg["chunk"]
        states, metrics = trainer(states, data, buf, jnp.int32(step), sub, ocfg["chunk"] * ocfg["updates_per_step"])
        if step % ocfg["eval_every"] == 0:
            jax.block_until_ready(states)
            evaluate_now(step, metrics)
    return states, log
