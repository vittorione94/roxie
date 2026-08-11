"""Sequence unpacking + TD-MPC world-model losses.

Covers `roxie.agents.utils.unpack_sequence`'s three masks against hand-computed
values at episode boundaries, then the model / policy losses on a real TOLD:
shapes, gradient flow, and the guarantee the masks exist for — that data past an
episode boundary cannot influence the loss. CPU only.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx, struct

from roxie.agents.utils import Transition, unpack_sequence
from roxie.losses.world_losses import tdmpc_model_loss_fn, tdmpc_policy_loss_fn
from roxie.models.actors import DeterministicActor
from roxie.models.critics import QCritic, TwinCritic
from roxie.models.world import TOLD, Encoder, LatentDynamics, RewardPredictor

GAMMA = 0.9
H = 3
OBS_DIM = 4
ACT_DIM = 2
LATENT_DIM = 6


@struct.dataclass
class _FakeSample:
    experience: Transition


@struct.dataclass
class _FakePair:
    first: Transition
    second: Transition


def _traj_sample(rewards, terminals, truncations, batch=1):
    """(B, T=H+1) trajectory sample; obs_j = j+1 so a selected obs is
    identifiable by value."""
    T = H + 1
    obs = jnp.tile(
        jnp.arange(1, T + 1, dtype=jnp.float32)[None, :, None], (batch, 1, OBS_DIM)
    )
    pad = lambda xs: jnp.tile(
        jnp.array([list(xs) + [0.0] * (T - len(xs))], dtype=jnp.float32), (batch, 1)
    )
    return _FakeSample(
        experience=Transition(
            observation=obs,
            action=jnp.zeros((batch, T, ACT_DIM)),
            reward=pad(rewards),
            terminal=pad(terminals).astype(bool),
            truncation=pad(truncations).astype(bool),
        )
    )


class TestUnpackSequence:
    def test_shapes(self):
        out = unpack_sequence(
            _traj_sample([1.0, 2.0, 3.0], [0, 0, 0], [0, 0, 0], batch=5), GAMMA, H
        )
        assert out["observations"].shape == (5, H + 1, OBS_DIM)
        assert out["actions"].shape == (5, H, ACT_DIM)
        assert out["rewards"].shape == (5, H)
        for key in ("reward_mask", "value_mask", "consistency_mask", "bootstrap"):
            assert out[key].shape == (5, H)

    def test_clean_window(self):
        out = unpack_sequence(
            _traj_sample([1.0, 2.0, 3.0], [0, 0, 0], [0, 0, 0]), GAMMA, H
        )
        np.testing.assert_allclose(out["reward_mask"][0], [1, 1, 1])
        np.testing.assert_allclose(out["value_mask"][0], [1, 1, 1])
        np.testing.assert_allclose(out["consistency_mask"][0], [1, 1, 1])
        np.testing.assert_allclose(out["bootstrap"][0], [GAMMA] * H, rtol=1e-6)
        np.testing.assert_allclose(out["rewards"][0], [1.0, 2.0, 3.0], rtol=1e-6)

    def test_terminal_mid_window(self):
        # terminal at j=1: that step's reward is real and its target is the
        # reward alone (bootstrap 0), but its stored successor is a reset state,
        # so consistency stops one step earlier. Nothing past j=1 is trainable.
        out = unpack_sequence(
            _traj_sample([1.0, 2.0, 99.0], [0, 1, 0], [0, 0, 0]), GAMMA, H
        )
        np.testing.assert_allclose(out["reward_mask"][0], [1, 1, 0])
        np.testing.assert_allclose(out["value_mask"][0], [1, 1, 0])
        np.testing.assert_allclose(out["consistency_mask"][0], [1, 0, 0])
        np.testing.assert_allclose(out["bootstrap"][0], [GAMMA, 0.0, GAMMA], rtol=1e-6)

    def test_truncation_mid_window(self):
        # truncation at j=1: reward is real, but the true successor is gone so
        # the step contributes to neither the TD target nor consistency.
        out = unpack_sequence(
            _traj_sample([1.0, 2.0, 99.0], [0, 0, 0], [0, 1, 0]), GAMMA, H
        )
        np.testing.assert_allclose(out["reward_mask"][0], [1, 1, 0])
        np.testing.assert_allclose(out["value_mask"][0], [1, 0, 0])
        np.testing.assert_allclose(out["consistency_mask"][0], [1, 0, 0])

    def test_terminal_on_first_step(self):
        out = unpack_sequence(
            _traj_sample([1.0, 99.0, 99.0], [1, 0, 0], [0, 0, 0]), GAMMA, H
        )
        np.testing.assert_allclose(out["reward_mask"][0], [1, 0, 0])
        np.testing.assert_allclose(out["value_mask"][0], [1, 0, 0])
        np.testing.assert_allclose(out["consistency_mask"][0], [0, 0, 0])
        assert out["bootstrap"][0, 0] == 0.0

    def test_terminal_wins_over_truncation(self):
        """Both flags on one step: treated as a terminal (no bootstrap)."""
        out = unpack_sequence(
            _traj_sample([1.0, 2.0, 3.0], [0, 1, 0], [0, 1, 0]), GAMMA, H
        )
        np.testing.assert_allclose(out["value_mask"][0], [1, 1, 0])
        assert out["bootstrap"][0, 1] == 0.0

    def test_rejects_flat_buffer(self):
        t = Transition(
            observation=jnp.zeros((1, OBS_DIM)),
            action=jnp.zeros((1, ACT_DIM)),
            reward=jnp.zeros((1,)),
            terminal=jnp.zeros((1,), dtype=bool),
            truncation=jnp.zeros((1,), dtype=bool),
        )
        with pytest.raises(ValueError, match="trajectory buffer"):
            unpack_sequence(
                _FakeSample(experience=_FakePair(first=t, second=t)), GAMMA, H
            )

    def test_rejects_too_short_window(self):
        sample = _traj_sample([1.0, 2.0, 3.0], [0, 0, 0], [0, 0, 0])
        with pytest.raises(ValueError, match="sample_sequence_length"):
            unpack_sequence(sample, GAMMA, H + 5)


@pytest.fixture
def told():
    return TOLD(
        encoder=Encoder(
            in_features=OBS_DIM,
            features=[32, 32],
            latent_dim=LATENT_DIM,
            rngs=nnx.Rngs(params=0, dropout=1),
        ),
        dynamics=LatentDynamics(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            features=[32, 32],
            rngs=nnx.Rngs(params=2, dropout=3),
        ),
        reward=RewardPredictor(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            features=[32, 32],
            rngs=nnx.Rngs(params=4, dropout=5),
        ),
        critic=TwinCritic(
            QCritic(
                in_features=LATENT_DIM + ACT_DIM,
                features=[32, 32],
                rngs=nnx.Rngs(params=6, dropout=7),
            ),
            QCritic(
                in_features=LATENT_DIM + ACT_DIM,
                features=[32, 32],
                rngs=nnx.Rngs(params=8, dropout=9),
            ),
        ),
    )


@pytest.fixture
def policy():
    return DeterministicActor(
        in_features=LATENT_DIM,
        features=[32, 32],
        action_dim=ACT_DIM,
        rngs=nnx.Rngs(params=10, dropout=11),
    )


def _target_of(model):
    import copy

    return copy.deepcopy(model)


def _loss_args():
    return dict(
        obs_mean=jnp.zeros(OBS_DIM),
        obs_std=jnp.ones(OBS_DIM),
        obs_clip=10.0,
        action_low=-jnp.ones(ACT_DIM),
        action_high=jnp.ones(ACT_DIM),
        rho=0.5,
        reward_coef=0.5,
        value_coef=0.1,
        consistency_coef=2.0,
    )


class TestModelLoss:
    def test_returns_finite_loss_and_aux(self, told, policy):
        samples = unpack_sequence(
            _traj_sample([1.0, 2.0, 3.0], [0, 0, 0], [0, 0, 0], batch=8), GAMMA, H
        )
        loss, aux = tdmpc_model_loss_fn(
            told, _target_of(told), policy, samples, **_loss_args()
        )
        assert jnp.isfinite(loss)
        assert aux["latents"].shape == (H, 8, LATENT_DIM)
        assert aux["policy_weights"].shape == (H, 8)
        for key in ("loss/reward", "loss/value", "loss/consistency"):
            assert jnp.isfinite(aux[key])

    def test_gradients_reach_every_component(self, told, policy):
        samples = unpack_sequence(
            _traj_sample([1.0, 2.0, 3.0], [0, 0, 0], [0, 0, 0], batch=8), GAMMA, H
        )
        target = _target_of(told)

        def loss_fn(model):
            loss, _ = tdmpc_model_loss_fn(
                model, target, policy, samples, **_loss_args()
            )
            return loss

        grads = nnx.grad(loss_fn)(told)
        flat = jax.tree.leaves(nnx.to_flat_state(grads))
        assert flat, "no gradients returned"
        assert all(jnp.all(jnp.isfinite(g)) for g in flat)

        # Every component must be reachable. The encoder in particular is only
        # reachable through the rolled latent chain, so a stray stop_gradient
        # anywhere in the rollout would show up here as an all-zero subtree.
        for component in ("encoder", "dynamics", "reward", "critic"):
            leaves = jax.tree.leaves(nnx.to_flat_state(grads[component]))
            assert leaves, f"no gradient leaves for {component}"
            assert any(
                jnp.any(jnp.abs(jnp.asarray(g)) > 0) for g in leaves
            ), f"{component} received only zero gradients"

    def test_masked_steps_cannot_influence_loss(self, told, policy):
        """The point of the masks: after a terminal at j=1, whatever sits in the
        buffer at j=2 belongs to another episode and must not move the loss."""
        target = _target_of(told)
        args = _loss_args()

        def loss_for(tail_reward):
            samples = unpack_sequence(
                _traj_sample([1.0, 2.0, tail_reward], [0, 1, 0], [0, 0, 0], batch=4),
                GAMMA,
                H,
            )
            loss, _ = tdmpc_model_loss_fn(told, target, policy, samples, **args)
            return loss

        np.testing.assert_allclose(loss_for(99.0), loss_for(-1234.0), rtol=1e-6)

    def test_all_masked_consistency_is_zero_not_nan(self, told, policy):
        """Terminal on step 0 leaves the consistency mask empty; the normalized
        mean must degrade to 0 rather than divide by zero."""
        samples = unpack_sequence(
            _traj_sample([1.0, 0.0, 0.0], [1, 0, 0], [0, 0, 0], batch=4), GAMMA, H
        )
        _, aux = tdmpc_model_loss_fn(
            told, _target_of(told), policy, samples, **_loss_args()
        )
        assert jnp.isfinite(aux["loss/consistency"])
        assert float(aux["loss/consistency"]) == 0.0

    def test_reward_head_fits_a_constant_reward(self, told, policy):
        """End-to-end sanity: a few Adam steps on a constant-reward, single-step
        window should drive the reward term down."""
        import optax

        samples = unpack_sequence(
            _traj_sample([1.0, 1.0, 1.0], [0, 0, 0], [0, 0, 0], batch=16), GAMMA, H
        )
        target = _target_of(told)
        args = dict(_loss_args())
        args.update(reward_coef=1.0, value_coef=0.0, consistency_coef=0.0)

        def loss_fn(model):
            loss, aux = tdmpc_model_loss_fn(
                model, target, policy, samples, **args
            )
            return loss, aux

        optimizer = nnx.Optimizer(told, optax.adam(1e-2), wrt=nnx.Param)
        _, aux0 = loss_fn(told)
        for _ in range(30):
            grads, _ = nnx.grad(loss_fn, has_aux=True)(told)
            optimizer.update(told, grads)
        _, aux1 = loss_fn(told)
        assert float(aux1["loss/reward"]) < float(aux0["loss/reward"])


class TestPolicyLoss:
    def test_finite_and_shaped(self, told, policy):
        samples = unpack_sequence(
            _traj_sample([1.0, 2.0, 3.0], [0, 0, 0], [0, 0, 0], batch=8), GAMMA, H
        )
        _, aux = tdmpc_model_loss_fn(
            told, _target_of(told), policy, samples, **_loss_args()
        )
        loss = tdmpc_policy_loss_fn(
            policy,
            told,
            aux["latents"],
            aux["policy_weights"],
            -jnp.ones(ACT_DIM),
            jnp.ones(ACT_DIM),
        )
        assert jnp.isfinite(loss)
        assert loss.shape == ()

    def test_gradient_flows_to_policy_only(self, told, policy):
        samples = unpack_sequence(
            _traj_sample([1.0, 2.0, 3.0], [0, 0, 0], [0, 0, 0], batch=8), GAMMA, H
        )
        _, aux = tdmpc_model_loss_fn(
            told, _target_of(told), policy, samples, **_loss_args()
        )
        latents, weights = aux["latents"], aux["policy_weights"]

        def loss_fn(pi):
            return tdmpc_policy_loss_fn(
                pi, told, latents, weights, -jnp.ones(ACT_DIM), jnp.ones(ACT_DIM)
            )

        grads = nnx.grad(loss_fn)(policy)
        flat = jax.tree.leaves(nnx.to_flat_state(grads))
        assert flat
        assert all(jnp.all(jnp.isfinite(g)) for g in flat)
        assert any(jnp.any(jnp.abs(g) > 0) for g in flat)

    def test_maximizes_q(self, told, policy):
        """A few steps of the policy loss should raise the mean Q it reports."""
        import optax

        samples = unpack_sequence(
            _traj_sample([1.0, 2.0, 3.0], [0, 0, 0], [0, 0, 0], batch=16), GAMMA, H
        )
        _, aux = tdmpc_model_loss_fn(
            told, _target_of(told), policy, samples, **_loss_args()
        )
        latents, weights = aux["latents"], aux["policy_weights"]

        def loss_fn(pi):
            return tdmpc_policy_loss_fn(
                pi, told, latents, weights, -jnp.ones(ACT_DIM), jnp.ones(ACT_DIM)
            )

        optimizer = nnx.Optimizer(policy, optax.adam(1e-2), wrt=nnx.Param)
        before = float(loss_fn(policy))
        for _ in range(20):
            optimizer.update(policy, nnx.grad(loss_fn)(policy))
        assert float(loss_fn(policy)) < before
