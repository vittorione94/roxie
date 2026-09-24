import jax
import jax.numpy as jnp

from roxie.exploration.noisy import (
    GaussianNoise,
    OrnsteinUhlenbeckNoise,
)
from roxie.exploration.schedulers import (
    ConstantSchedule,
    LinearDecay,
    ExponentialDecay,
)


class TestSchedules:
    """Each schedule is one closed-form curve; one test walks each of them from
    start to past the end rather than asserting a point per cell."""

    def test_constant_never_moves(self):
        schedule = ConstantSchedule()
        assert schedule(1.0, 0) == 1.0
        assert schedule(1.0, 1000) == 1.0
        assert schedule(0.5, 500) == 0.5

    def test_linear_interpolates_then_holds_the_final_value(self):
        schedule = LinearDecay(decay_steps=100, final_value=0.0)
        assert jnp.isclose(schedule(1.0, 0), 1.0)
        assert jnp.isclose(schedule(1.0, 50), 0.5)
        assert jnp.isclose(schedule(1.0, 100), 0.0)
        # Past the end it holds, rather than continuing downward.
        assert jnp.isclose(LinearDecay(decay_steps=100, final_value=0.1)(1.0, 200), 0.1)

    def test_exponential_halves_per_decay_step_and_floors(self):
        schedule = ExponentialDecay(decay_rate=0.5, decay_steps=100)
        assert jnp.isclose(schedule(1.0, 0), 1.0)
        assert jnp.isclose(schedule(1.0, 100), 0.5)
        assert jnp.isclose(
            ExponentialDecay(decay_rate=0.1, decay_steps=10, min_value=0.05)(1.0, 100),
            0.05,
        )


class TestGaussianNoise:
    def test_samples_at_the_action_shape_or_a_given_one(self, rng_key):
        noise = GaussianNoise(action_shape=(4,))
        assert noise.sample_noise(rng_key).shape == (4,)
        # The batched call the rollout makes: one draw per parallel env.
        assert noise.sample_noise(rng_key, shape=(8, 4)).shape == (8, 4)

    def test_evaluation_is_a_no_op_that_does_not_advance_the_schedule(self, rng_key):
        """Eval must neither perturb the action nor age the anneal — an epoch's
        eval would otherwise decay exploration every epoch."""
        noise = GaussianNoise(action_shape=(4,), initial_noise_scale=0.5)
        actions = jnp.ones((4,), dtype=jnp.float32)

        assert jnp.allclose(noise.add_noise(actions, rng_key, evaluation=True), actions)
        assert noise.step_count.get_value() == 0

    def test_training_perturbs_and_counts_the_step(self, rng_key):
        noise = GaussianNoise(action_shape=(4,), initial_noise_scale=0.5)
        actions = jnp.ones((4,), dtype=jnp.float32)

        assert not jnp.allclose(noise.add_noise(actions, rng_key, evaluation=False), actions)
        assert noise.step_count.get_value() == 1

    def test_reports_its_hyperparameters_and_its_current_scale(self):
        schedule = LinearDecay(decay_steps=100, final_value=0.0)
        noise = GaussianNoise(
            action_shape=(4,), initial_noise_scale=1.0, decay_schedule=schedule
        )
        hp = noise.hyperparameters()
        assert hp["name"] == "Gaussian"
        assert hp["initial_noise_scale"] == 1.0
        assert hp["action_shape"] == (4,)
        assert jnp.isclose(noise.get_current_scale(), 1.0)


class TestOrnsteinUhlenbeckNoise:
    def test_draws_are_shaped_correlated_and_resettable(self, rng_key):
        """OU's whole point is a state that carries between draws — so it must
        walk away from where it started, and `reset_noise` must return it."""
        noise = OrnsteinUhlenbeckNoise(action_shape=(4,), theta=0.15)

        first = noise.sample_noise(rng_key)
        assert first.shape == (4,)
        for i in range(1, 10):
            last = noise.sample_noise(jax.random.fold_in(rng_key, i))
        assert not jnp.allclose(first, last)

        assert not jnp.allclose(noise.noise_state[...], jnp.zeros((4,), jnp.float32))
        noise.reset_noise()
        assert jnp.allclose(noise.noise_state[...], jnp.zeros((4,), jnp.float32))

    def test_reports_its_hyperparameters(self):
        noise = OrnsteinUhlenbeckNoise(
            action_shape=(4,), initial_noise_scale=0.3, theta=0.2, mu=0.1
        )
        hp = noise.hyperparameters()
        assert hp["name"] == "Ornstein-Uhlenbeck"
        assert hp["theta"] == 0.2
        assert hp["mu"] == 0.1
