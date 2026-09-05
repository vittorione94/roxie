"""`pop_diagnostics` is a contract every learning agent honours, identically.

It used to be an optional hook the trainer duck-typed with `getattr`, and only
TD3 and PPO implemented it — so the five other agents ran blind on exactly the
metrics TD3's tuning turns on. TD4 was the sharpest case: it is TD3 plus a
distributional critic, but it inherits from D4PG, so it reported nothing about
the tanh saturation that `pre_activation_coef` exists to fight.

This file is what keeps that from coming back. It drives each agent from its own
shipped config through a real rollout and a real update — the diagnostics are
built inside `lax.scan` (and, under a `policy_delay`, inside a nested
`nnx.cond`), where a missing key or a mismatched pytree fails at trace time
rather than at review time.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

import roxie.agents  # noqa: F401  (avoid circular import)
from roxie.agents.utils import build_agent
from roxie.environment.vector import Timestep

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "roxie" / "configs" / "agent"

OBS, ACT, ENVS = 8, 3, 16
# Rows of `ENVS` transitions to buffer before the first update, and in total.
# The warmup is not optional: flashbax allocates with `jnp.empty_like`, so an
# update that fires before the buffer holds `min_length` samples reads
# uninitialized memory and every value metric comes back at ~1e8 or NaN.
WARMUP_ROWS, ROLLOUT_STEPS = 12, 32

# Keys shared by every actor that squashes through a tanh, from `actor_aux`.
SQUASHED_ACTOR = {"pre_act_abs", "pre_act_max", "sat_frac", "tanh_grad", "actor_q"}
# Keys shared by every critic that regresses on a bootstrap, from `value_aux`.
BOOTSTRAPPED_CRITIC = {"q_buffer", "q_target", "td_abs"}
DETERMINISTIC = SQUASHED_ACTOR | BOOTSTRAPPED_CRITIC | {
    "pre_act_penalty", "target_smooth_clip_frac", "target_act_rail_frac",
}
TWIN = {"twin_gap"}

# What each agent must report, beyond the `updates_per_env_step` /
# `buffer_frac` levels the base class adds. Spelled out per agent rather than
# derived, so dropping a metric is a test failure and not a quiet regression.
EXPECTED = {
    "ddpg": DETERMINISTIC,
    "d4pg": DETERMINISTIC,
    "td3": DETERMINISTIC | TWIN,
    "td4": DETERMINISTIC | TWIN,
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
    },
}

# Bounded by construction; a value outside says the reduction divided by the
# wrong denominator (the classic being `n_steps` where the actor only ran on
# `n_steps / policy_delay` of them).
FRACTIONS = {
    "sat_frac", "tanh_grad", "target_smooth_clip_frac", "target_act_rail_frac",
    "clip_frac", "buffer_frac",
}


def _build(name: str):
    """The agent its shipped config builds, with the buffer and the update
    schedule shrunk to what a unit test can drive."""
    cfg = OmegaConf.create(
        {
            "env": {"parallel_envs": ENVS},
            "agent": OmegaConf.load(CONFIG_DIR / f"{name}.yaml"),
        }
    ).agent

    memory = cfg.get("memory_config", None)
    if memory is not None:
        for key, value in (("max_length", 64 * ENVS), ("min_length", 8 * ENVS),
                           ("max_length_time_axis", 32), ("sample_batch_size", 16),
                           # PPO's queue hardcodes the env count rather than
                           # reading `${env.parallel_envs}`.
                           ("add_batch_size", ENVS)):
            if key in memory:
                memory[key] = value
        if "sample_sequence_length" in memory:
            memory.sample_sequence_length = min(memory.sample_sequence_length, 8)

    # Learn as soon as the buffer is legitimately full, then on every step: the
    # point is to exercise the burst, not the schedule (`test_update_schedule.py`
    # covers that).
    warmup = WARMUP_ROWS * ENVS
    for key, value in (("steps_between_updates", ENVS),
                       ("memory_warmup", warmup), ("learning_steps", 2)):
        if key in cfg:
            cfg[key] = value

    kwargs = dict(
        env_obs_size=OBS,
        env_action_size=ACT,
        action_low=jnp.full((ACT,), -1.0),
        action_high=jnp.full((ACT,), 1.0),
    )
    if "noise_config" in cfg or name in ("ddpg", "d4pg", "td3", "td4"):
        kwargs["noise_config"] = OmegaConf.load(
            REPO / "roxie" / "configs" / "noise" / "gaussian.yaml"
        )
    return build_agent(cfg, **kwargs)


def _run(agent, steps=ROLLOUT_STEPS) -> int:
    """One epoch's worth of the trainer's loop: act, buffer, update.

    Returns the env-step count the trainer would hand `pop_diagnostics`.
    """
    key = jax.random.PRNGKey(0)
    obs = jax.random.normal(jax.random.PRNGKey(1), (ENVS, OBS))
    false = jnp.zeros((ENVS,), jnp.bool_)
    env_steps = 0

    for _ in range(steps):
        key, act_key, obs_key, rew_key, upd_key = jax.random.split(key, 5)
        agent.step(obs, evaluate=False, key=act_key)
        timestep = Timestep(
            obs=jax.random.normal(obs_key, (ENVS, OBS)),
            reward=jax.random.normal(rew_key, (ENVS,)),
            terminated=false,
            truncated=false,
            info={},
        )
        agent.add(obs, timestep)
        obs = timestep.obs
        env_steps += ENVS
        agent.update(steps=env_steps, agent_rng=upd_key)

    return env_steps


@pytest.fixture(scope="module")
def diagnostics():
    """One drained epoch per agent, built once — each is a real compile."""
    out = {}
    for name in EXPECTED:
        agent = _build(name)
        assert agent.pop_diagnostics() == {}, (
            f"{name} reported diagnostics before running a single update"
        )
        env_steps = _run(agent)
        out[name] = (agent, agent.pop_diagnostics(env_steps), env_steps)
    return out


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_agent_reports_its_own_metrics(name, diagnostics):
    _agent, diag, _steps = diagnostics[name]
    missing = {f"{name}/{key}" for key in EXPECTED[name]} - set(diag)
    assert not missing, f"{name} stopped reporting {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_value_is_a_finite_float_under_the_agents_own_prefix(name, diagnostics):
    _agent, diag, _steps = diagnostics[name]
    assert diag, f"{name} ran updates but reported nothing"
    for key, value in diag.items():
        assert key.startswith(f"{name}/"), key
        assert isinstance(value, float), key
        assert jnp.isfinite(value), key


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_bounded_metrics_stay_bounded(name, diagnostics):
    """A fraction above 1 means the epoch reduction used the wrong denominator."""
    _agent, diag, _steps = diagnostics[name]
    for key, value in diag.items():
        if key.split("/", 1)[1] in FRACTIONS:
            assert 0.0 <= value <= 1.0, f"{key} = {value}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_realized_replay_ratio_is_reported(name, diagnostics):
    """The metric that would have caught the release_v1 walker grid, where six
    off-policy arms ran 5M env steps at zero gradient steps."""
    _agent, diag, _steps = diagnostics[name]
    assert diag[f"{name}/updates_per_env_step"] > 0.0


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_replay_fill_is_reported_by_exactly_the_replay_driven_agents(
    name, diagnostics
):
    agent, diag, _steps = diagnostics[name]
    reported = f"{name}/buffer_frac" in diag
    assert reported == (agent.buffer_size is not None)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_second_read_in_the_same_epoch_reports_nothing(name, diagnostics):
    """Draining is what keeps an epoch's mean from carrying the last one's
    bursts; a gap is honest, a stale repeat is not."""
    agent, _diag, env_steps = diagnostics[name]
    assert agent.pop_diagnostics(env_steps) == {}
