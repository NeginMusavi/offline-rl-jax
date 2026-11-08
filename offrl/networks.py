"""Small MLP networks as pure JAX functions.

A network is an ``init`` function that returns parameters (a plain pytree: a list of
``{"W", "b"}`` dicts) and an ``apply`` function that takes those parameters as input.
"""
import jax
import jax.numpy as jnp
import numpy as np

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
_HALF_LOG_2PI = 0.9189385332046727


def init_linear(key, n_in, n_out):
    """Uniform in +-1/sqrt(fan_in) for weights and biases (PyTorch's nn.Linear default)."""
    kw, kb = jax.random.split(key)
    bound = 1.0 / np.sqrt(n_in)
    return {"W": jax.random.uniform(kw, (n_in, n_out), minval=-bound, maxval=bound),
            "b": jax.random.uniform(kb, (n_out,), minval=-bound, maxval=bound)}


def init_mlp(key, sizes):
    """sizes = [in, hidden, ..., out]"""
    keys = jax.random.split(key, len(sizes) - 1)
    return [init_linear(k, i, o) for k, i, o in zip(keys, sizes[:-1], sizes[1:])]


def mlp(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["W"] + layer["b"])
    return x @ params[-1]["W"] + params[-1]["b"]


# --- the four networks used by BC and IQL ---------------------------------------------------

def policy_mean(pi, s):
    """Deterministic action in [-1, 1]: BC's policy, and IQL's action at test time."""
    return jnp.tanh(mlp(pi["net"], s))


def policy_std(pi):
    """State-independent standard deviation of IQL's Gaussian policy."""
    return jnp.exp(jnp.clip(pi["log_std"], LOG_STD_MIN, LOG_STD_MAX))


def gaussian_log_prob(pi, s, a):
    """log N(a; mean(s), std), summed over action dimensions."""
    mean, log_std = policy_mean(pi, s), jnp.clip(pi["log_std"], LOG_STD_MIN, LOG_STD_MAX)
    return jnp.sum(-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std - _HALF_LOG_2PI, -1)


def value(v, s):
    """V(s)"""
    return mlp(v, s)[..., 0]


def twin_q(q, s, a):
    """(Q1(s, a), Q2(s, a))"""
    sa = jnp.concatenate([s, a], -1)
    return mlp(q["q1"], sa)[..., 0], mlp(q["q2"], sa)[..., 0]
