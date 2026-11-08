"""Implicit Q-Learning (Kostrikov, Nair & Levine, 2021, https://arxiv.org/abs/2110.06169).

Each update does three things, never evaluating actions that are not in the data:

1. V(s):    expectile regression toward min(Q1, Q2)_target(s, a) of dataset actions
            L_V = E[ |tau - 1(Q - V < 0)| (Q - V)^2 ]
2. Q(s, a): TD learning with V(s') instead of a max over actions
            L_Q = E[ (r + gamma (1 - d) V(s') - Q(s, a))^2 ]   (for both Q networks)
3. policy:  advantage-weighted behavior cloning
            L_pi = -E[ min(exp(beta (Q - V)), max_weight) log pi(a | s) ]

Each loss only reaches its own network (the advantage is stop-gradient'ed for the policy and
the TD target uses stop_gradient(V(s'))), so one gradient of the summed loss equals three
separate gradients. The Q target uses V(s') from before this step's V update, as in CORL.
"""
import jax
import jax.numpy as jnp
import optax

from ..networks import init_mlp, value, twin_q, gaussian_log_prob


def make_iql(cfg, s_dim, a_dim):
    """Returns (init, update, fresh_opts), all pure functions (see bc.make_bc).

    cfg keys: hidden_dim, lr, discount, tau, expectile, beta, max_weight, iql_steps,
              actor_lr_schedule ("cosine": decay to 0 over iql_steps, as in the IQL paper;
                                 "constant": used for online fine-tuning).
    """
    h, lr = cfg["hidden_dim"], cfg["lr"]
    opt_q, opt_v = optax.adam(lr), optax.adam(lr)
    if cfg.get("actor_lr_schedule", "cosine") == "constant":
        opt_pi = optax.adam(lr)
    else:
        opt_pi = optax.adam(optax.cosine_decay_schedule(lr, cfg["iql_steps"]))

    def init(key):
        k = jax.random.split(key, 4)
        q = {"q1": init_mlp(k[0], [s_dim + a_dim, h, h, 1]), "q2": init_mlp(k[1], [s_dim + a_dim, h, h, 1])}
        v = init_mlp(k[2], [s_dim, h, h, 1])
        pi = {"net": init_mlp(k[3], [s_dim, h, h, a_dim]), "log_std": jnp.zeros(a_dim)}
        return {"q": q, "q_target": q, "v": v, "pi": pi,
                "opt_q": opt_q.init(q), "opt_v": opt_v.init(v), "opt_pi": opt_pi.init(pi)}

    def loss_fn(nets, q_target, b):
        q, v, pi = nets["q"], nets["v"], nets["pi"]
        target_q = jnp.minimum(*twin_q(q_target, b["s"], b["a"]))
        td_target = b["r"] + cfg["discount"] * (1 - b["d"]) * jax.lax.stop_gradient(value(v, b["s2"]))
        # 1) V
        v_s = value(v, b["s"])
        adv = target_q - v_s
        v_loss = jnp.mean(jnp.abs(cfg["expectile"] - (adv < 0)) * adv ** 2)
        # 2) Q
        q1, q2 = twin_q(q, b["s"], b["a"])
        q_loss = jnp.mean((q1 - td_target) ** 2) + jnp.mean((q2 - td_target) ** 2)
        # 3) policy
        w = jnp.minimum(jnp.exp(cfg["beta"] * jax.lax.stop_gradient(adv)), cfg["max_weight"])
        pi_loss = -jnp.mean(w * gaussian_log_prob(pi, b["s"], b["a"]))
        metrics = {"q_loss": q_loss, "v_loss": v_loss, "pi_loss": pi_loss, "v_mean": v_s.mean()}
        return v_loss + q_loss + pi_loss, metrics

    def update(state, b):
        nets = {"q": state["q"], "v": state["v"], "pi": state["pi"]}
        grads, metrics = jax.grad(loss_fn, has_aux=True)(nets, state["q_target"], b)
        new = dict(state)
        for name, opt in [("q", opt_q), ("v", opt_v), ("pi", opt_pi)]:
            upd, new["opt_" + name] = opt.update(grads[name], state["opt_" + name], state[name])
            new[name] = optax.apply_updates(state[name], upd)
        new["q_target"] = optax.incremental_update(new["q"], state["q_target"], cfg["tau"])
        return new, metrics

    def fresh_opts(state):
        return {**state, "opt_q": opt_q.init(state["q"]), "opt_v": opt_v.init(state["v"]),
                "opt_pi": opt_pi.init(state["pi"])}

    return init, update, fresh_opts
