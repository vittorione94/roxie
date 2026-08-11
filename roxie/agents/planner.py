"""MPPI / CEM planning in the TD-MPC latent space (Hansen et al. 2022).

Action selection is a short trajectory optimization rather than a policy
forward pass: candidate action sequences are rolled through the learned latent
dynamics, scored by predicted rewards plus a terminal Q, and the sampling
distribution is refit to the best of them over a few iterations. Only the first
action of the winning sequence is executed, and the fitted mean is carried over
to warm-start the next step.

Everything is written with an explicit leading candidate axis, `(N, B, ...)` for
N candidates across B environments — the world-model nets broadcast over leading
axes, so no `vmap` over model parameters is needed. Both loops (horizon and CEM
iterations) are `lax.scan`s, so the whole planner is one compiled program with
no host round-trips, matching how the rest of the repo dispatches to the GPU.

Conventions: candidate actions live in [-1, 1] and are scaled to env units
before they reach the model (the buffer stores env-unit actions, so that is what
the model was trained on).
"""

import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent


def _policy_action(policy_model, z, key, std):
    """Policy prior action at latent `z`, in [-1, 1], perturbed by `std`.

    The repo's `DeterministicActor` is deterministic, so without this noise the
    policy-seeded candidate trajectories would be N identical copies. TD-MPC
    likewise samples its policy prior with a small fixed std.
    """
    a = policy_model(z)
    a = a + std * jax.random.normal(key, a.shape)
    return jnp.clip(a, -1.0, 1.0)


def estimate_value(
    model,
    policy_model,
    z,
    actions,
    key,
    gamma,
    min_std,
    action_low,
    action_high,
):
    """Score action sequences by imagined return.

    `z` is (N, B, L), `actions` is (H, N, B, A) in [-1, 1]. Returns (N, B):
    ``sum_t gamma^t R(z_t, a_t) + gamma^H min_i Q_i(z_H, pi(z_H))``.

    The terminal Q is what keeps a short horizon honest — it stands in for all
    reward beyond H, so the planner optimizes the full discounted return rather
    than just the next H steps.
    """

    def body(z, a_t):
        a_env = Agent.scale_to_env(a_t, action_low, action_high)
        z, reward = model.step(z, a_env)
        return z, jnp.squeeze(reward, -1)

    z_final, rewards = jax.lax.scan(body, z, actions)  # rewards (H, N, B)

    # Horizon is static (it is the scanned axis), so the discounts are built
    # outside the loop rather than carried through it.
    horizon = actions.shape[0]
    discounts = gamma ** jnp.arange(horizon, dtype=jnp.float32)
    imagined_return = jnp.sum(discounts[:, None, None] * rewards, axis=0)

    terminal_action = _policy_action(policy_model, z_final, key, min_std)
    q1, q2 = model.q(
        z_final, Agent.scale_to_env(terminal_action, action_low, action_high)
    )
    terminal_q = jnp.minimum(jnp.squeeze(q1, -1), jnp.squeeze(q2, -1))

    return imagined_return + (gamma ** horizon) * terminal_q


def _policy_trajectories(
    model, policy_model, z, key, horizon, min_std, action_low, action_high
):
    """Roll the policy prior forward to seed part of the candidate set.

    Gives the optimizer a few trajectories that are already decent, which
    matters most under an iteration budget this small; pure Gaussian sampling in
    an H*A-dimensional space would rarely find them on its own. Returns
    (H, N_pi, B, A) in [-1, 1].
    """

    def body(z, step_key):
        a = _policy_action(policy_model, z, step_key, min_std)
        z = model.next(z, Agent.scale_to_env(a, action_low, action_high))
        return z, a

    _, actions = jax.lax.scan(body, z, jax.random.split(key, horizon))
    return actions


def plan(
    model,
    policy_model,
    z,
    prev_mean,
    key,
    action_low,
    action_high,
    *,
    horizon: int,
    num_samples: int = 256,
    num_elites: int = 32,
    num_policy_trajectories: int = 24,
    num_iterations: int = 6,
    gamma: float = 0.99,
    temperature: float = 0.5,
    momentum: float = 0.1,
    min_std: float = 0.05,
    max_std: float = 2.0,
    evaluate: bool = False,
):
    """Plan one action per environment by MPPI in the latent space.

    This is the *un-jitted* body. Call it when you are already inside a trace
    (e.g. from the trainer's compiled eval rollout); use `plan_jit` for
    top-level eager calls such as the acting loop. Nesting `nnx.jit` inside
    another trace fails on the models' rng state ("cannot mutate RngCount from a
    different trace level"), which is why the two entry points are separate.

    Args:
        model: the `TOLD` world model.
        policy_model: the policy prior, used to seed candidates and to supply
            the terminal action for the bootstrap Q.
        z: (B, L) current latents (the caller encodes; the planner never sees
            raw observations).
        prev_mean: (B, H, A) warm start from the previous step, in [-1, 1].
        key: PRNG key.
        action_low / action_high: env action bounds, for scaling candidates into
            the units the model was trained on.

    Returns:
        `(action, next_mean, std, noise)` — action (B, A) in [-1, 1] ready for
        `Agent.scale_to_env`, the shifted plan to carry into the next step, the
        mean first-step std as a scalar convergence diagnostic, and the
        exploration noise actually applied (post-clip), in the same
        clean-minus-executed convention as `Agent.deterministic_step_fn`.
    """
    batch = z.shape[0]
    action_dim = prev_mean.shape[-1]

    key, pi_key = jax.random.split(key)
    policy_actions = _policy_trajectories(
        model,
        policy_model,
        jnp.broadcast_to(z, (num_policy_trajectories,) + z.shape),
        pi_key,
        horizon,
        min_std,
        action_low,
        action_high,
    )  # (H, N_pi, B, A)

    # Every candidate starts from the same current latent.
    z_candidates = jnp.broadcast_to(
        z, (num_samples + num_policy_trajectories, batch, z.shape[-1])
    )

    # `prev_mean` warm-starts the search from last step's plan; std starts wide.
    mean = jnp.swapaxes(prev_mean, 0, 1)  # (H, B, A)
    std = jnp.full((horizon, batch, action_dim), max_std, dtype=jnp.float32)

    # The elites/scores of the LAST iteration are what gets executed, so they
    # ride in the carry rather than being stacked as scan outputs — stacking
    # would hold num_iterations copies of an (H, k, B, A) array for nothing.
    init_elites = jnp.zeros((horizon, num_elites, batch, action_dim), jnp.float32)
    init_score = jnp.zeros((num_elites, batch), jnp.float32)

    def cem_iteration(carry, iter_key):
        mean, std, _, _ = carry
        sample_key, value_key = jax.random.split(iter_key)

        # Gaussian candidates around the current mean, plus the policy seeds.
        noise = jax.random.normal(sample_key, (horizon, num_samples, batch, action_dim))
        gaussian = jnp.clip(mean[:, None] + std[:, None] * noise, -1.0, 1.0)
        actions = jnp.concatenate([gaussian, policy_actions], axis=1)  # (H, N, B, A)

        values = estimate_value(
            model,
            policy_model,
            z_candidates,
            actions,
            value_key,
            gamma,
            min_std,
            action_low,
            action_high,
        )  # (N, B)

        # Top-k per environment. lax.top_k works on the trailing axis, so the
        # candidate axis is moved there and back.
        elite_values, elite_idx = jax.lax.top_k(
            jnp.swapaxes(values, 0, 1), num_elites
        )  # both (B, k)
        elite_values = jnp.swapaxes(elite_values, 0, 1)  # (k, B)
        elite_idx = jnp.swapaxes(elite_idx, 0, 1)  # (k, B)
        elite_actions = jnp.take_along_axis(
            actions, elite_idx[None, :, :, None], axis=1
        )  # (H, k, B, A)

        # Softmax weights over the elites, shifted by the max for stability.
        score = jnp.exp(temperature * (elite_values - jnp.max(elite_values, axis=0)))
        score = score / (jnp.sum(score, axis=0) + 1e-9)  # (k, B)

        w = score[None, :, :, None]  # (1, k, B, 1)
        new_mean = jnp.sum(w * elite_actions, axis=1)
        new_std = jnp.sqrt(
            jnp.sum(w * jnp.square(elite_actions - new_mean[:, None]), axis=1)
        )
        new_std = jnp.clip(new_std, min_std, max_std)

        # Momentum on the mean only (as in the reference implementation): the
        # std must stay free to collapse, since it is the convergence signal.
        mean = momentum * mean + (1.0 - momentum) * new_mean
        return (mean, new_std, elite_actions, score), None

    (mean, std, elite_actions, score), _ = jax.lax.scan(
        cem_iteration,
        (mean, std, init_elites, init_score),
        jax.random.split(key, num_iterations),
    )

    # Execute one elite's plan: sampled in proportion to its score while
    # training (extra exploration), the best one when evaluating. TD-MPC samples
    # in both cases; taking the argmax at eval keeps scoring deterministic, in
    # line with how the other agents here evaluate.
    key, choice_key, noise_key = jax.random.split(key, 3)
    if evaluate:
        chosen = jnp.argmax(score, axis=0)  # (B,)
    else:
        chosen = jax.random.categorical(choice_key, jnp.log(score + 1e-9), axis=0)

    executed = jnp.take_along_axis(elite_actions, chosen[None, None, :, None], axis=1)
    executed = executed[:, 0]  # (H, B, A)

    planned = executed[0]
    action = planned
    if not evaluate:
        action = action + std[0] * jax.random.normal(noise_key, action.shape)
    action = jnp.clip(action, -1.0, 1.0)

    # Warm start for the next step: shift the fitted plan one step forward and
    # repeat its last entry, so the next search starts from the part of this
    # plan that has not been executed yet.
    next_mean = jnp.concatenate([mean[1:], mean[-1:]], axis=0)

    return action, jnp.swapaxes(next_mean, 0, 1), jnp.mean(std[0]), planned - action


# Compiled entry point for top-level (eager) callers — the acting loop. Inside
# an existing trace, call `plan` directly instead.
plan_jit = nnx.jit(
    plan,
    static_argnames=(
        "horizon",
        "num_samples",
        "num_elites",
        "num_policy_trajectories",
        "num_iterations",
        "evaluate",
    ),
)
