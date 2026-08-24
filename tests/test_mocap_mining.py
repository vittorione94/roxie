"""Negative mining over mocap start phases.

The point of the feature is that reset actually stops sampling uniformly, so
these tests check the SAMPLED START DISTRIBUTION, not just that the bookkeeping
arithmetic is self-consistent.

Mining is the mocap env adapting its OWN ``params``: it is written entirely
against the driver-agnostic ``FuncEnv`` hooks (``init_params`` /
``observe_params`` / ``epoch_refresh``), and roxie itself knows nothing about
it. These call those hooks directly, exactly as the rollout does.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

BINS = 16


@pytest.fixture(scope="module")
def env():
    loader = pytest.importorskip("examples.mocap.loader")
    cfg = loader.load_default_config()
    cfg.negative_mining.enabled = True
    cfg.negative_mining.bins = BINS
    cfg.negative_mining.alpha = 0.9   # strong, so the shift is unambiguous
    cfg.negative_mining.ema = 0.0     # no smoothing: one refresh reaches target
    train_env, _test_env, _ = loader.load_mocap_env(
        config=cfg, clip_ids=["CMU_006_13"], impl="jax"
    )
    return train_env


def _table(env, fail_bin=None, visits=100.0, fails=50.0):
    """`params` as it looks part-way through an epoch."""
    params = env.init_params()
    params["visit"] = jnp.full((BINS,), visits)
    if fail_bin is not None:
        params["fail"] = params["fail"].at[fail_bin].add(fails)
    return params


def _refresh(env, params):
    """The weights `epoch_refresh` produces, minus the clip-swap flag."""
    refreshed, _invalidated = env.epoch_refresh(params)
    return refreshed


class TestMiningRefresh:
    def test_starts_uniform(self, env):
        params = env.init_params()
        assert np.allclose(np.asarray(params["weights"]), 1.0 / BINS)
        assert float(jnp.sum(params["fail"])) == 0.0

    def test_an_env_without_mining_has_no_params(self, env):
        """The hook is the generic one, so "off" must mean None — the same
        thing every other env in the repo hands the driver."""
        assert env._mining_enabled  # guard: the fixture turned it ON
        env._mining_enabled = False
        try:
            assert env.init_params() is None
            assert env.observe_params(None, {}, jnp.array([True])) is None
            assert env.epoch_refresh(None) == (None, False)
        finally:
            env._mining_enabled = True

    def test_concentrates_on_the_failing_bin(self, env):
        target = 5
        out = _refresh(env, _table(env, fail_bin=target))
        new_w = np.asarray(out["weights"])
        assert int(np.argmax(new_w)) == target
        assert new_w[target] > 5 * new_w[(target + 3) % BINS]
        assert np.isclose(new_w.sum(), 1.0)
        # Counters must reset, or difficulty would integrate over the whole run
        # and stop tracking the CURRENT policy.
        assert float(jnp.sum(out["fail"])) == 0.0
        assert float(jnp.sum(out["visit"])) == 0.0

    def test_no_failures_stays_uniform(self, env):
        """Nothing failing must not be mistaken for 'everything is hard'."""
        out = _refresh(env, _table(env, fail_bin=None))
        assert np.allclose(np.asarray(out["weights"]), 1.0 / BINS, atol=1e-6)

    def test_uses_rate_not_raw_count(self, env):
        """A bin visited 10x less but failing every time is HARDER, not easier.

        Raw counts would rank it below a heavily-visited bin with a few
        failures, which is exactly backwards: dying early is why the later
        phases have few visits in the first place.
        """
        params = _table(env, visits=1000.0)
        params["visit"] = params["visit"].at[3].set(10.0)
        params["fail"] = params["fail"].at[3].set(10.0)    # rate 1.0, count 10
        params["fail"] = params["fail"].at[7].set(100.0)   # rate 0.1, count 100
        out = _refresh(env, params)
        assert int(np.argmax(np.asarray(out["weights"]))) == 3


class TestMiningObserve:
    def test_clip_end_is_not_a_failure(self, env):
        """Truncation means the reference ran out, not that tracking failed.

        The driver feeds `timestep.terminated` (done minus truncation), so a
        truncated episode contributes a VISIT but not a FAIL — otherwise the end
        of every clip looks maximally hard and soaks up the start budget.
        """
        params = env.init_params()
        info = {"phase_idx": jnp.array([10, 10]), "clip_len": jnp.array([160, 160])}
        # env 0 truncated (termination=False), env 1 genuinely failed.
        out = env.observe_params(params, info, jnp.array([False, True]))
        assert float(jnp.sum(out["visit"])) == 2.0
        assert float(jnp.sum(out["fail"])) == 1.0

    def test_observing_leaves_the_weights_alone(self, env):
        """Only `epoch_refresh` moves the start distribution; a step may only
        count. Weights that drifted per step would change the reset
        distribution mid-epoch, under a reset pool built before it."""
        params = env.init_params()
        info = {"phase_idx": jnp.array([10]), "clip_len": jnp.array([160])}
        out = env.observe_params(params, info, jnp.array([True]))
        assert np.array_equal(np.asarray(out["weights"]),
                              np.asarray(params["weights"]))

    def test_bins_by_clip_relative_phase(self, env):
        params = env.init_params()
        info = {"phase_idx": jnp.array([0, 159]), "clip_len": jnp.array([160, 160])}
        out = env.observe_params(params, info, jnp.array([True, True]))
        f = np.asarray(out["fail"])
        assert f[0] == 1.0 and f[BINS - 1] == 1.0
        assert f.sum() == 2.0


class TestSampledStarts:
    """The behaviour that actually matters: does reset follow the weights?"""

    @staticmethod
    def _phases(env, params, n=256):
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        if params is None:
            states = jax.vmap(env.reset)(keys)
        else:
            states = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
        return np.asarray(states.info["phase_idx"])

    def test_uniform_weights_spread_over_the_clip(self, env):
        p = self._phases(env, env.init_params())
        assert len(np.unique(p)) > 50, "uniform weights should spread starts"

    def test_mined_weights_shift_the_starts(self, env):
        """Concentrate difficulty in one bin; starts must follow it."""
        target = 5
        mined = _refresh(env, _table(env, fail_bin=target))

        p_uniform = self._phases(env, env.init_params())
        p_mined = self._phases(env, mined)

        # Fraction of starts landing in the target bin's frame range.
        hi = int(p_uniform.max()) + 1
        lo_f = target / BINS
        hi_f = (target + 1) / BINS

        def frac_in_bin(p):
            return float(np.mean((p >= lo_f * hi) & (p < hi_f * hi)))

        assert frac_in_bin(p_mined) > 3 * max(frac_in_bin(p_uniform), 1e-3), (
            f"mined {frac_in_bin(p_mined):.3f} vs uniform {frac_in_bin(p_uniform):.3f}"
        )
        # ...but coverage must NOT collapse: this is re-weighting, not a curriculum.
        assert len(np.unique(p_mined)) > 20

    def test_reset_without_params_still_works(self, env):
        """Backward compatibility: the second argument is optional."""
        p = self._phases(env, None)
        assert p.shape == (256,)
        assert len(np.unique(p)) > 50


class TestMiningMetrics:
    """The diagnostics ride out with the ordinary per-step metrics, so the
    driver needs no mining-shaped channel to carry them."""

    def test_effective_bins_falls_as_weights_concentrate(self, env):
        uniform = env.init_params()
        mined = _refresh(env, _table(env, fail_bin=5))
        s_uniform = env._mining_metrics(uniform)
        s_mined = env._mining_metrics(mined)
        assert float(s_uniform["mining/effective_bins"]) == pytest.approx(
            BINS, rel=1e-3
        )
        assert float(s_mined["mining/effective_bins"]) < BINS
        assert float(s_mined["mining/max_weight_ratio"]) > 1.0
        assert int(s_mined["mining/hardest_bin"]) == 5

    def test_transition_info_carries_them_only_with_params(self, env):
        """No params (every other env, and the eval driver) => no mining keys,
        so the metric dict an env emits never depends on a hook it lacks."""
        keys = jax.random.split(jax.random.PRNGKey(0), 1)
        state = jax.vmap(env.reset)(keys)
        state = jax.tree.map(lambda x: x[0], state)
        action = jnp.zeros(env.action_size)

        plain = env.transition_info(state, action, state, None)
        assert not any(k.startswith("mining/") for k in plain["metrics"])

        mined = env.transition_info(state, action, state, env.init_params())
        assert "mining/effective_bins" in mined["metrics"]
        # The env's own metrics must survive alongside them.
        assert set(plain["metrics"]) < set(mined["metrics"])
