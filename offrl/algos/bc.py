"""Behavior cloning: regress the dataset action, ||pi(s) - a||^2."""
import jax
import jax.numpy as jnp
import optax

from ..networks import init_mlp, policy_mean


def make_bc(cfg, s_dim, a_dim):
    """Returns (init, update, fresh_opts), all pure functions.

    init(key) -> state                       networks + optimizer state
    update(state, batch) -> (state, metrics) one gradient step
    fresh_opts(state) -> state               same networks, new optimizer state
    """
    h = cfg["hidden_dim"]
    opt = optax.adam(cfg["lr"])

    def init(key):
        pi = {"net": init_mlp(key, [s_dim, h, h, a_dim])}
        return {"pi": pi, "opt": opt.init(pi)}

    def loss_fn(pi, b):
        return jnp.mean((policy_mean(pi, b["s"]) - b["a"]) ** 2)

    def update(state, b):
        loss, grads = jax.value_and_grad(loss_fn)(state["pi"], b)
        upd, opt_state = opt.update(grads, state["opt"], state["pi"])
        return {"pi": optax.apply_updates(state["pi"], upd), "opt": opt_state}, {"loss": loss}

    def fresh_opts(state):
        return {**state, "opt": opt.init(state["pi"])}

    return init, update, fresh_opts
