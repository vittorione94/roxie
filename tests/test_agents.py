"""Every learning agent's shared contract, driven through its shipped config.

Each agent is built, warmed and taught ONCE for this module, and every test
below reads that one result — a build plus a learning pass is a compile, and
there are seven agents. What is pinned is what the trainer relies on from all of
them alike: a pass trains, the targets track, the diagnostics arrive namespaced
and finite,
and acting is bounded always and deterministic under `evaluate`.

A real pass rather than unit tests of the loss functions, because the pass is
a `lax.scan` — under a `policy_delay`, `nnx.cond` nested inside one — which is
exactly where a missing key or a mismatched pytree fails at trace time rather
than at review time.

What one algorithm adds on top (SAC's temperature, MPO's duals, the delayed
policy update, the deterministic arms' saturation penalty) is at the bottom,
one test each.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests import harness
from tests.harness import AGENTS, DETERMINISTIC, ENVS, OBS

# Keys shared by every actor that squashes through a tanh, from `actor_aux`.
SQUASHED_ACTOR = {"pre_act_abs", "pre_act_max", "sat_frac", "tanh_grad", "actor_q"}
# Keys shared by every critic that regresses on a bootstrap, from `value_aux`.
BOOTSTRAPPED_CRITIC = {"q_buffer", "q_target", "td_abs"}
DPG = SQUASHED_ACTOR | BOOTSTRAPPED_CRITIC | {
    "pre_act_penalty", "target_smooth_clip_frac", "target_act_rail_frac",
}
TWIN = {"twin_gap"}

# What each agent must report, beyond the `updates_per_env_step` / `buffer_frac`
# levels the base class adds. Spelled out per agent rather than derived, so
# dropping a metric is a test failure and not a quiet regression.
EXPECTED = {
    "ddpg": DPG,
    "d4pg": DPG,
    "td3": DPG | TWIN,
    "td4": DPG | TWIN,
    "sac": SQUASHED_ACTOR | BOOTSTRAPPED_CRITIC | TWIN | {
        "alpha", "entropy", "target_act_rail_frac",
    },
    "mpo": SQUASHED_ACTOR | BOOTSTRAPPED_CRITIC | {
        "policy_loss", "temperature_loss", "kl_loss", "temperature",
        "kl_mean", "kl_stddev", "target_act_rail_frac",
    },
    "ppo": {
        "approx_kl", "clip_frac", "kl_early_stops",
        "steps_per_rollout", "epochs_per_rollout",
        "entropy", "policy_std", "sat_frac", "tanh_grad",
        "adv_abs", "adv_scale", "value_ev", "trunc_frac", "term_frac", "reward_abs",
    },
}

# Bounded by construction; a value outside says the reduction divided by the
# wrong denominator (the classic being `n_steps` where the actor only ran on
# `n_steps / policy_delay` of them).
FRACTIONS = {
    "sat_frac", "tanh_grad", "target_smooth_clip_frac", "target_act_rail_frac",
    "clip_frac", "buffer_frac",
}

# The networks every agent carries, and the target copies only a replay-driven
# one does.
NETWORKS = ("actor", "critic")
TARGETS = ("target_actor", "target_critic")


@dataclasses.dataclass
class LearningPass:
    """One agent's first learning pass, and what it was before it."""

    agent: Any
    before: dict
    steps: int
    actor_loss: Any
    critic_loss: Any
    diagnostics: dict
    env_steps: int


def _networks(agent) -> dict:
    return {
        name: getattr(agent.state, name)
        for name in NETWORKS + TARGETS
        if getattr(agent.state, name, None) is not None
    }


@pytest.fixture(scope="module")
def passes():
    """One warmed agent and one drained pass per agent — each a real compile."""
    out = {}
    for name in AGENTS:
        agent, env_steps = harness.warmed(name)
        assert agent.pop_diagnostics() == {}, (
            f"{name} reported diagnostics before running a single update — a "
            f"misleading zero is worse than a gap in the curve"
        )
        before = {
            key: harness.snapshot(module)
            for key, module in _networks(agent).items()
        }
        steps, actor_loss, critic_loss = agent.learn(jax.random.PRNGKey(2))
        out[name] = LearningPass(
            agent=agent,
            before=before,
            steps=int(steps),
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            diagnostics=agent.pop_diagnostics(env_steps),
            env_steps=env_steps,
        )
    return out


@pytest.mark.parametrize("name", AGENTS)
def test_a_learning_pass_trains_every_network_it_carries(name, passes):
    """Losses finite, gradient steps actually taken, and both networks moved.

    The targets are the separate claim: their soft update runs inside the scan
    body, so a pass of N steps must move them N times — not once, and not zero
    times, which is what a target closed over instead of carried would do.
    """
    learning = passes[name]
    assert learning.steps > 0, "no gradient step ran — every assertion here is vacuous"
    assert jnp.isfinite(learning.actor_loss) and jnp.isfinite(learning.critic_loss)

    for key, module in _networks(learning.agent).items():
        assert harness.moved(learning.before[key], module), (
            f"{name}: {key} is unchanged after a learning pass"
        )


@pytest.mark.parametrize("name", AGENTS)
def test_diagnostics_arrive_namespaced_finite_and_bounded(name, passes):
    """`pop_diagnostics` is a contract every learning agent honours identically:
    its own metrics, each a finite host float under its own prefix, and the
    ones bounded by construction inside their bounds."""
    diag = passes[name].diagnostics
    assert diag, f"{name} ran updates but reported nothing"

    missing = {f"{name}/{key}" for key in EXPECTED[name]} - set(diag)
    assert not missing, f"{name} stopped reporting {sorted(missing)}"

    for key, value in diag.items():
        assert key.startswith(f"{name}/"), key
        assert isinstance(value, float), key
        assert np.isfinite(value), key
        if key.rsplit("/", 1)[-1] in FRACTIONS:
            assert 0.0 <= value <= 1.0, f"{key} = {value} is not a fraction"

    # The realized replay ratio, which is what says the schedule fired at all.
    assert diag[f"{name}/updates_per_env_step"] > 0.0
    # `buffer_frac` is the replay-driven agents' alone: PPO has a queue it
    # drains every rollout, and a fill fraction would read as a full buffer.
    assert (f"{name}/buffer_frac" in diag) is (name != "ppo")


@pytest.mark.parametrize("name", AGENTS)
def test_a_second_read_in_the_same_epoch_reports_nothing(name, passes):
    """The drain is what stops an epoch that ran no pass from re-reporting the
    last one's numbers."""
    assert passes[name].agent.pop_diagnostics(passes[name].env_steps) == {}


@pytest.mark.parametrize("name", AGENTS)
def test_acting_is_bounded_stochastic_and_deterministic_under_evaluate(name, passes):
    """Three claims about `select_action`, which every rollout depends on:
    exploration draws differ between keys, an eval does not, and neither leaves
    the action space."""
    agent = passes[name].agent
    obs = jax.random.normal(jax.random.PRNGKey(3), (ENVS, OBS), dtype=jnp.float32)

    explored, _noise, _extras = agent.select_action(obs, jax.random.PRNGKey(0))
    other, _noise, _extras = agent.select_action(obs, jax.random.PRNGKey(1))
    assert not jnp.allclose(explored, other), "exploration ignores the key"
    assert jnp.all(explored >= -1.0) and jnp.all(explored <= 1.0)

    evaluated, applied_noise, _extras = agent.select_action(
        obs, jax.random.PRNGKey(0), evaluate=True
    )
    again, _noise, _extras = agent.select_action(
        obs, jax.random.PRNGKey(1), evaluate=True
    )
    assert jnp.allclose(evaluated, again), "eval actions depend on the key"
    assert jnp.allclose(applied_noise, 0.0), "eval reported exploration noise"
    assert jnp.all(evaluated >= -1.0) and jnp.all(evaluated <= 1.0)


@pytest.mark.parametrize("name", AGENTS)
def test_consecutive_passes_survive_the_donated_state(name, passes):
    """Every `_<agent>_grad_steps` donates its train state. An agent that read the
    donated (now invalid) input rather than rebinding `self.state` to the result
    keeps working for exactly one pass, so this runs two more.
    """
    agent = passes[name].agent
    for i in range(2):
        # PPO's queue was consumed by the pass that drained it, so each of its
        # passes needs a rollout of its own; the replay agents keep theirs.
        harness.drive(agent, harness.WARMUP_ITERS, jax.random.PRNGKey(10 + i))
        steps, actor_loss, critic_loss = agent.learn(jax.random.PRNGKey(20 + i))
        assert steps > 0
        assert jnp.isfinite(actor_loss) and jnp.isfinite(critic_loss)

    # This is the one test that advances a shared agent; draining what it banked
    # leaves the module's fixture as every other test expects to find it,
    # whatever order they run in.
    agent.pop_diagnostics()


# What one algorithm adds on top


def test_sac_temperature_rides_in_the_scan_carry():
    """`log_alpha` and its Adam slots live in the scan carry. If they were
    closed over instead, alpha would advance by one step per pass no matter how
    long the pass is — this pins the difference. `auto_alpha: false` is the
    same knob read the other way: the temperature must not move at all."""
    alphas = {}
    for steps in (1, 8):
        agent, _ = harness.warmed("sac", learning_steps=steps)
        agent.learn(jax.random.PRNGKey(0))
        alphas[steps] = float(agent.log_alpha_module.log_alpha[...])

    assert alphas[1] != pytest.approx(0.0), "alpha did not update at all"
    assert abs(alphas[8]) > abs(alphas[1]), (
        f"an 8-step pass moved alpha to {alphas[8]}, no further than a 1-step "
        f"pass {alphas[1]}: the temperature is not carried through the scan"
    )

    frozen, _ = harness.warmed("sac", auto_alpha=False, init_log_alpha=-1.0)
    frozen.learn(jax.random.PRNGKey(0))
    assert float(frozen.log_alpha_module.log_alpha[...]) == pytest.approx(-1.0)


def test_mpo_duals_ride_in_the_scan_carry():
    """The temperature and the two decoupled KL multipliers are MPO's Lagrange
    duals; they and their Adam slots ride in the carry, with the same failure
    mode as SAC's temperature above."""
    temperatures = {}
    for steps in (1, 8):
        agent, _ = harness.warmed("mpo", learning_steps=steps)
        agent.learn(jax.random.PRNGKey(0))
        temperatures[steps] = float(
            np.ravel(np.asarray(agent.dual_params.log_temperature[...]))[0]
        )

    initial = float(
        np.ravel(np.asarray(harness.build("mpo").dual_params.log_temperature[...]))[0]
    )
    assert temperatures[1] != pytest.approx(initial), "temperature never updated"
    assert abs(temperatures[8] - initial) > abs(temperatures[1] - initial), (
        f"an 8-step pass moved the temperature to {temperatures[8]}, no further "
        f"than a 1-step pass {temperatures[1]}: the duals are not carried"
    )


def test_policy_delay_gates_the_actor_but_never_the_critic():
    """The actor update runs under `nnx.cond` inside the scan. With a delay
    longer than the pass only the first step updates the actor — but every step
    must still update the critic, or `policy_delay` would be a learning-rate
    knob rather than a delay."""
    agent, _ = harness.warmed("td3", learning_steps=4, policy_delay=99)
    before = {
        key: harness.snapshot(getattr(agent.state, key))
        for key in ("actor", "critic")
    }
    agent.learn(jax.random.PRNGKey(0))

    assert harness.moved(before["critic"], agent.state.critic), (
        "the critic must update on every step regardless of policy_delay"
    )
    # Step 0 is always an update step, so the actor still moved once.
    assert harness.moved(before["actor"], agent.state.actor)


def test_policy_delay_does_not_leak_zeros_into_the_actor_metrics():
    """The skipped `nnx.cond` branch returns zeros to keep the pytree shape, so
    `_<agent>_grad_steps` must divide by the number of ACTOR updates and not by
    `n_steps` — otherwise every actor metric reads `1/policy_delay` of its true
    value."""
    agent, _ = harness.warmed("td3", learning_steps=2, policy_delay=2)

    agent.learn(jax.random.PRNGKey(0), n_steps=2)  # 1 actor update out of 2
    delayed = agent.pop_diagnostics()["td3/tanh_grad"]
    agent.learn(jax.random.PRNGKey(0), n_steps=1)  # 1 actor update out of 1
    undelayed = agent.pop_diagnostics()["td3/tanh_grad"]

    # Both are means over exactly one actor update on a near-identical actor:
    # within a hair of each other, and nowhere near a 2x dilution.
    assert delayed == pytest.approx(undelayed, rel=0.15)


@pytest.mark.parametrize("name", DETERMINISTIC)
def test_the_saturation_penalty_reaches_every_deterministic_actor(name):
    """`pre_activation_coef` has to be live on all four deterministic arms.

    `experiments/dmc/agent/ddpg_bench.yaml` calls them "a clean algorithm-only
    A/B" on the strength of them carrying the same coefficient, and an arm whose
    actor loss quietly ignored the argument would still round-trip the value
    through the config, the checkpoint and the hyperparameter log. Hence testing
    it by DIFFERENTIATION: the same pass at two coefficients must leave the
    actor in different places.

    The hinge is at |u| = 1 and the shipped actor starts at `output_init_scale`
    0.01, so an untouched actor emits |u| ~ 0.1 and the penalty is identically
    zero — the bias is what puts the logits where the knob can bite.
    """
    def trained(coef):
        agent = harness.build(name, pre_activation_coef=coef)
        for module in (agent.state.actor, agent.state.target_actor):
            module.output_layer.bias[...] = jnp.full(
                (harness.ACT,), 3.0, dtype=jnp.float32
            )
        harness.drive(agent, harness.WARMUP_ITERS, jax.random.PRNGKey(0))
        agent.learn(jax.random.PRNGKey(1))
        return harness.snapshot(agent.state.actor)

    off, on = trained(0.0), trained(100.0)
    gap = max(float(np.max(np.abs(a - b))) for a, b in zip(off, on))
    assert gap > 1e-6, (
        f"{name}: pre_activation_coef 0 and 100 trained the actor to the same "
        f"weights, so the knob is dead for this agent"
    )
