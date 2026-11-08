"""Core math: networks, BC, IQL (the merged update equals three separate updates)."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.stats import norm

from offrl.algos.bc import make_bc
from offrl.algos.iql import make_iql
from offrl.networks import gaussian_log_prob, init_mlp, mlp, policy_mean, twin_q, value

CFG = dict(hidden_dim=32, lr=3e-4, iql_steps=50, discount=0.99, tau=0.005, expectile=0.9, beta=10.0,
           max_weight=100.0, actor_lr_schedule="cosine")
S, A, B = 7, 3, 64


def batch(seed):
    r = np.random.default_rng(seed)
    return dict(s=jnp.asarray(r.normal(size=(B, S)), jnp.float32), a=jnp.asarray(r.uniform(-1, 1, (B, A)), jnp.float32),
                r=jnp.asarray((r.random(B) < 0.1) - 1.0, jnp.float32), s2=jnp.asarray(r.normal(size=(B, S)), jnp.float32),
                d=jnp.asarray(r.random(B) < 0.05, jnp.float32))


def test_mlp_shapes_and_init_range():
    p = init_mlp(jax.random.PRNGKey(0), [S, 16, 16, A])
    assert mlp(p, jnp.zeros((5, S))).shape == (5, A)
    w = np.asarray(p[0]["W"])
    assert np.abs(w).max() <= 1 / np.sqrt(S) + 1e-6          # PyTorch's nn.Linear default range


def test_gaussian_log_prob_matches_scipy():
    pi = {"net": init_mlp(jax.random.PRNGKey(1), [S, 16, A]), "log_std": jnp.array([-1.0, 0.0, 0.5])}
    b = batch(0)
    mean = policy_mean(pi, b["s"])
    ref = norm.logpdf(b["a"], mean, jnp.exp(pi["log_std"])).sum(-1)
    np.testing.assert_allclose(gaussian_log_prob(pi, b["s"], b["a"]), ref, rtol=1e-5, atol=1e-5)


def test_bc_reduces_loss():
    init, update, _ = make_bc(CFG, S, A)
    st = init(jax.random.PRNGKey(0)); b = batch(1)
    upd = jax.jit(update)
    first = float(upd(st, b)[1]["loss"])
    for _ in range(200):
        st, m = upd(st, b)
    assert float(m["loss"]) < 0.8 * first          # clearly decreasing on a fixed batch


def reference_iql_step(st, b, opts):
    """Three separate gradient steps (V, Q, policy), all from the parameters before the step."""
    opt_q, opt_v, opt_pi = opts
    target_q = jnp.minimum(*twin_q(st["q_target"], b["s"], b["a"]))
    next_v = value(st["v"], b["s2"])
    adv_old = target_q - value(st["v"], b["s"])

    def v_loss(v):
        adv = target_q - value(v, b["s"])
        return jnp.mean(jnp.abs(CFG["expectile"] - (adv < 0)) * adv ** 2)

    def q_loss(q):
        td = b["r"] + CFG["discount"] * (1 - b["d"]) * next_v
        q1, q2 = twin_q(q, b["s"], b["a"])
        return jnp.mean((q1 - td) ** 2) + jnp.mean((q2 - td) ** 2)

    def pi_loss(pi):
        w = jnp.minimum(jnp.exp(CFG["beta"] * adv_old), CFG["max_weight"])
        return -jnp.mean(w * gaussian_log_prob(pi, b["s"], b["a"]))

    new = dict(st)
    for name, fn, opt in [("v", v_loss, opt_v), ("q", q_loss, opt_q), ("pi", pi_loss, opt_pi)]:
        upd, new["opt_" + name] = opt.update(jax.grad(fn)(st[name]), st["opt_" + name], st[name])
        new[name] = optax.apply_updates(st[name], upd)
    new["q_target"] = jax.tree.map(lambda t, q: (1 - CFG["tau"]) * t + CFG["tau"] * q, st["q_target"], new["q"])
    return new


def test_iql_merged_update_equals_separate_updates():
    init, update, _ = make_iql(CFG, S, A)
    st = init(jax.random.PRNGKey(0)); ref = st
    opts = (optax.adam(CFG["lr"]), optax.adam(CFG["lr"]), optax.adam(optax.cosine_decay_schedule(CFG["lr"], CFG["iql_steps"])))
    upd = jax.jit(update)
    for i in range(20):
        b = batch(i)
        st, _ = upd(st, b)
        ref = reference_iql_step(ref, b, opts)
    for k in ["q", "q_target", "v", "pi"]:
        for x, y in zip(jax.tree.leaves(st[k]), jax.tree.leaves(ref[k])):
            np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)


def test_fresh_opts_restart_and_constant_schedule():
    init, update, _ = make_iql(CFG, S, A)
    st = init(jax.random.PRNGKey(0))
    for i in range(5):
        st, _ = update(st, batch(i))
    _, update_c, fresh = make_iql({**CFG, "actor_lr_schedule": "constant"}, S, A)
    st2 = fresh(st)
    assert all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(st["pi"]), jax.tree.leaves(st2["pi"])))
    counts = [np.asarray(l) for l in jax.tree.leaves(st2["opt_pi"]) if np.ndim(l) == 0]
    assert all(int(c) == 0 for c in counts)                   # optimizer state restarted
    st3, _ = update_c(st2, batch(9))
    assert not all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(st2["pi"]), jax.tree.leaves(st3["pi"])))
