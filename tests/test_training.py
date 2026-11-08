"""Training loops: seeds train independently; mixed offline/online sampling; the online buffer."""
import jax
import jax.numpy as jnp
import numpy as np

from offrl.algos.finetune import buffer_insert, empty_online_buffer
from offrl.algos.iql import make_iql
from offrl.training import make_mixed_trainer, make_trainer, sample_mixed_batch, seed_keys

CFG = dict(hidden_dim=16, lr=3e-4, iql_steps=100, discount=0.99, tau=0.005, expectile=0.9, beta=3.0, max_weight=100.0)
S, A = 5, 2


def data(n, fill):
    return dict(s=jnp.full((n, S), fill, jnp.float32), s2=jnp.full((n, S), fill, jnp.float32),
                a=jnp.zeros((n, A)), r=jnp.full((n,), -1.0), d=jnp.zeros((n,)))


def test_vmapped_seeds_train_independently():
    init, update, _ = make_iql(CFG, S, A)
    states = jax.vmap(init)(seed_keys([0, 1, 2], 0))
    r = np.random.default_rng(0)
    d = dict(s=jnp.asarray(r.normal(size=(500, S)), jnp.float32), s2=jnp.asarray(r.normal(size=(500, S)), jnp.float32),
             a=jnp.asarray(r.uniform(-1, 1, (500, A)), jnp.float32), r=-jnp.ones(500), d=jnp.zeros(500))
    states, metrics = make_trainer(update, 32)(states, d, seed_keys([0, 1, 2], 1), 20)
    assert np.asarray(metrics["q_loss"]).shape == (3,)
    w = np.asarray(states["pi"]["net"][0]["W"])
    assert not np.allclose(w[0], w[1])                          # different seeds -> different networks


def test_mixed_sampling_fractions():
    off, on = data(1000, 0.0), data(1000, 1.0)
    for size, expected in [(0, 0.0), (1000, 0.5), (250, 0.2)]:
        b = sample_mixed_batch(off, on, jnp.int32(size), jax.random.PRNGKey(size), 20000)
        frac_online = float(b["s"][:, 0].mean())
        assert abs(frac_online - expected) < 0.02, (size, frac_online)


def test_mixed_trainer_runs():
    init, update, _ = make_iql(CFG, S, A)
    states = jax.vmap(init)(seed_keys([0, 1], 0))
    on = jax.tree.map(lambda x: jnp.stack([x, x]), data(100, 1.0))
    states, m = make_mixed_trainer(update, 16)(states, data(200, 0.0), on, jnp.int32(50), seed_keys([0, 1], 1), 10)
    assert np.isfinite(np.asarray(m["q_loss"])).all()


def test_buffer_insert():
    buf = empty_online_buffer(2, 10, S, A)
    new = jax.tree.map(lambda x: jnp.ones((2, 4) + x.shape[2:]), buf)
    buf = buffer_insert(buf, new, 4)
    s = np.asarray(buf["s"][:, :, 0])
    assert (s[:, 4:8] == 1).all() and (s[:, :4] == 0).all() and (s[:, 8:] == 0).all()
