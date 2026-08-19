"""Negative mining over mocap start phases.

The point of the feature is that reset actually stops sampling uniformly, so
these tests check the SAMPLED START DISTRIBUTION, not just that the bookkeeping
arithmetic is self-consistent.
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


@pytest.fixture(scope="module")
def env():
    loader = pytest.importorskip("examples.mocap.loader")
    cfg = loader.load_default_config()
    cfg.negative_mining.enabled = True
    cfg.negative_mining.bins = 16
    cfg.negative_mining.alpha = 0.9   # strong, so the shift is unambiguous
    cfg.negative_mining.ema = 0.0     # no smoothing: one refresh reaches target
    train_env, _test_env, _ = loader.load_mocap_env(
        config=cfg, clip_ids=["CMU_006_13"], impl="jax"
    )
    return train_env.env  # unwrap TerminationWrapper


def _counts(env, fail_bin=None, visits=100.0, fails=50.0):
    n = env.mining_bins
    c = {"visit": jnp.full((n,), visits), "fail": jnp.zeros((n,))}
    if fail_bin is not None:
        c["fail"] = c["fail"].at[fail_bin].add(fails)
    return c


class TestMiningRefresh:
    def test_starts_uniform(self, env):
        w, counts = env.mining_init()
        assert np.allclose(np.asarray(w), 1.0 / env.mining_bins)
        assert float(jnp.sum(counts["fail"])) == 0.0

    def test_concentrates_on_the_failing_bin(self, env):
        w, _ = env.mining_init()
        target = 5
        new_w, zeroed = env.mining_refresh(w, _counts(env, fail_bin=target))
        new_w = np.asarray(new_w)
        assert int(np.argmax(new_w)) == target
        assert new_w[target] > 5 * new_w[(target + 3) % env.mining_bins]
        assert np.isclose(new_w.sum(), 1.0)
        # Counters must reset, or difficulty would integrate over the whole run
        # and stop tracking the CURRENT policy.
        assert float(jnp.sum(zeroed["fail"])) == 0.0
        assert float(jnp.sum(zeroed["visit"])) == 0.0

    def test_no_failures_stays_uniform(self, env):
        """Nothing failing must not be mistaken for 'everything is hard'."""
        w, _ = env.mining_init()
        new_w, _ = env.mining_refresh(w, _counts(env, fail_bin=None))
        assert np.allclose(np.asarray(new_w), 1.0 / env.mining_bins, atol=1e-6)

    def test_uses_rate_not_raw_count(self, env):
        """A bin visited 10x less but failing every time is HARDER, not easier.

        Raw counts would rank it below a heavily-visited bin with a few
        failures, which is exactly backwards: dying early is why the later
        phases have few visits in the first place.
        """
        w, _ = env.mining_init()
        c = {"visit": jnp.full((env.mining_bins,), 1000.0),
             "fail": jnp.zeros((env.mining_bins,))}
        c["visit"] = c["visit"].at[3].set(10.0)
        c["fail"] = c["fail"].at[3].set(10.0)    # rate 1.0, count 10
        c["fail"] = c["fail"].at[7].set(100.0)   # rate 0.1, count 100
        new_w, _ = env.mining_refresh(w, c)
        assert int(np.argmax(np.asarray(new_w))) == 3


class TestMiningObserve:
    def test_clip_end_is_not_a_failure(self, env):
        """Truncation means the reference ran out, not that tracking failed.

        The trainer feeds `info["termination"]` (done minus truncation), so a
        truncated episode contributes a VISIT but not a FAIL — otherwise the end
        of every clip looks maximally hard and soaks up the start budget.
        """
        _, counts = env.mining_init()
        info = {"phase_idx": jnp.array([10, 10]), "clip_len": jnp.array([160, 160])}
        # env 0 truncated (termination=False), env 1 genuinely failed.
        out = env.mining_observe(counts, info, jnp.array([False, True]))
        assert float(jnp.sum(out["visit"])) == 2.0
        assert float(jnp.sum(out["fail"])) == 1.0

    def test_bins_by_clip_relative_phase(self, env):
        _, counts = env.mining_init()
        n = env.mining_bins
        info = {"phase_idx": jnp.array([0, 159]), "clip_len": jnp.array([160, 160])}
        out = env.mining_observe(counts, info, jnp.array([True, True]))
        f = np.asarray(out["fail"])
        assert f[0] == 1.0 and f[n - 1] == 1.0
        assert f.sum() == 2.0


class TestSampledStarts:
    """The behaviour that actually matters: does reset follow the weights?"""

    @staticmethod
    def _phases(env, weights, n=256):
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        if weights is None:
            states = jax.vmap(env.reset)(keys)
        else:
            states = jax.vmap(env.reset, in_axes=(0, None))(keys, weights)
        return np.asarray(states.info["phase_idx"])

    def test_uniform_weights_spread_over_the_clip(self, env):
        w, _ = env.mining_init()
        p = self._phases(env, w)
        assert len(np.unique(p)) > 50, "uniform weights should spread starts"

    def test_mined_weights_shift_the_starts(self, env):
        """Concentrate difficulty in one bin; starts must follow it."""
        w, _ = env.mining_init()
        target = 5
        mined, _ = env.mining_refresh(w, _counts(env, fail_bin=target))

        p_uniform = self._phases(env, w)
        p_mined = self._phases(env, mined)

        # Fraction of starts landing in the target bin's frame range.
        hi = int(p_uniform.max()) + 1
        lo_f = target / env.mining_bins
        hi_f = (target + 1) / env.mining_bins

        def frac_in_bin(p):
            return float(np.mean((p >= lo_f * hi) & (p < hi_f * hi)))

        assert frac_in_bin(p_mined) > 3 * max(frac_in_bin(p_uniform), 1e-3), (
            f"mined {frac_in_bin(p_mined):.3f} vs uniform {frac_in_bin(p_uniform):.3f}"
        )
        # ...but coverage must NOT collapse: this is re-weighting, not a curriculum.
        assert len(np.unique(p_mined)) > 20

    def test_reset_without_weights_still_works(self, env):
        """Backward compatibility: the second argument is optional."""
        p = self._phases(env, None)
        assert p.shape == (256,)
        assert len(np.unique(p)) > 50


class TestMiningStats:
    def test_effective_bins_falls_as_weights_concentrate(self, env):
        w, _ = env.mining_init()
        mined, _ = env.mining_refresh(w, _counts(env, fail_bin=5))
        s_uniform = env.mining_stats(w, _counts(env))
        s_mined = env.mining_stats(mined, _counts(env, fail_bin=5))
        assert float(s_uniform["mining/effective_bins"]) == pytest.approx(
            env.mining_bins, rel=1e-3
        )
        assert float(s_mined["mining/effective_bins"]) < env.mining_bins
        assert float(s_mined["mining/max_weight_ratio"]) > 1.0
        assert int(s_mined["mining/hardest_bin"]) == 5
