"""Some basic non-learning agents used for example for debugging."""

from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from roxie.agents.agent import Agent


class NormalRandom(Agent):
    """Random agent producing actions from normal distributions."""

    def __init__(
        self,
        env_obs_size,
        env_action_size,
        action_low,
        action_high,
        seed=0,
        loc=0,
        scale=1,
    ):
        self.loc = loc
        self.scale = scale
        self.action_size = env_action_size
        self.action_low = action_low
        self.action_high = action_high
        self.np_random = np.random.RandomState(seed)

    def step(self, observations, steps):
        return self._policy(observations)

    def test_step(self, observations, steps):
        return self._policy(observations)

    def _policy(self, observations):
        return self.np_random.normal(self.loc, self.scale, self.action_size)

    def _export_hyperparams(self) -> Dict[str, Any]:
        return {}


class UniformRandom(Agent):
    """Random agent producing actions from uniform distributions."""

    def initialize(self, observation_space, action_size, seed=None):
        self.action_size = action_size
        self.np_random = np.random.RandomState(seed)

    def step(self, observations, steps):
        return self._policy(observations)

    def test_step(self, observations, steps):
        return self._policy(observations)

    def _policy(self, observations):
        return self.np_random.uniform(-1, 1, self.action_size)


class OrnsteinUhlenbeck(Agent):
    """Non-learning baseline that drives the env with a temporally correlated
    Ornstein-Uhlenbeck process instead of a learned policy.

    Useful as a sanity-check / exploration baseline: it conforms to the same
    agent contract the ``Trainer`` and ``play`` loops expect (``step`` /
    ``add`` / ``update`` / ``save`` / ``load`` with batched JAX actions in env
    units), but never learns — ``update`` is a no-op. The OU state is kept per
    env and decorrelated back to zero when an episode resets.
    """

    def __init__(
        self,
        env_obs_size: int,
        env_action_size: int,
        action_low: jnp.ndarray,
        action_high: jnp.ndarray,
        *,
        scale: float = 0.2,
        theta: float = 0.15,
        dt: float = 1e-2,
        clip: float = 2.0,
        mu: float = 0.0,
        seed: int = 0,
    ):
        self.action_size = int(env_action_size)
        self.action_low = action_low
        self.action_high = action_high
        self.scale = scale
        self.theta = theta
        self.dt = dt
        self.clip = clip
        self.mu = mu
        self.seed = int(seed)

        # Per-env OU state in [-1, 1], lazily sized on the first ``step`` once
        # the batch (number of parallel envs) is known.
        self.actions = None
        # Folded into the externally supplied key each step so the noise still
        # advances when the caller passes a constant key (e.g. the play loop).
        self._t = 0
        # No observation normalization / replay warmup for this baseline; the
        # Trainer branches on these attributes.
        self.normalize_observations = False

        print("OrnsteinUhlenbeck agent initialized.")
        print("Hyper Params:", self._export_hyperparams())

    def step(
        self,
        observation: jnp.ndarray,
        evaluate: bool = False,
        key: jax.Array = None,
    ) -> jnp.ndarray:
        batch = observation.shape[0]
        if self.actions is None or self.actions.shape[0] != batch:
            self.actions = jnp.zeros((batch, self.action_size))

        step_key = jax.random.fold_in(key, self._t)
        self._t += 1

        noise = jnp.clip(jax.random.normal(step_key, self.actions.shape),
                         -self.clip, self.clip)
        actions = self.actions + self.theta * self.dt * (self.mu - self.actions)
        actions = actions + self.scale * jnp.sqrt(self.dt) * noise
        self.actions = jnp.clip(actions, -1.0, 1.0)

        self.last_action = Agent.scale_to_env(
            self.actions, self.action_low, self.action_high
        )
        return self.last_action

    def add(self, prev_obs, timestep):
        # Non-learning: nothing to store. Decorrelate the OU state for any env
        # whose episode just ended so a fresh episode starts from zero noise.
        if self.actions is not None:
            done = jnp.logical_or(timestep.terminated, timestep.truncated)
            keep = (1.0 - done.astype(self.actions.dtype))[:, None]
            self.actions = self.actions * keep

    def update(self, steps, agent_rng):
        # No gradients, ever.
        return 0, 0, 0

    def _export_hyperparams(self) -> Dict[str, Any]:
        return {
            "scale": float(self.scale),
            "theta": float(self.theta),
            "dt": float(self.dt),
            "clip": float(self.clip),
            "mu": float(self.mu),
            "seed": int(self.seed),
            "env_action_size": int(self.action_size),
        }

    # --- Checkpointing -------------------------------------------------------
    # There are no learned params; persist just the hyperparameters and the
    # action bounds so ``play`` can rebuild an identical agent from a run dir.
    def save(self, path, *, extra_metadata: Dict[str, Any] = None, **_):
        path = Path(path).resolve()
        payload = {
            "format_version": 1,
            "hyperparams": self._export_hyperparams(),
            "action_low": np.asarray(jax.device_get(self.action_low)),
            "action_high": np.asarray(jax.device_get(self.action_high)),
            "metadata": (extra_metadata or {}),
        }
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path, payload)
        checkpointer.wait_until_finished()
        print(f"[OrnsteinUhlenbeck.save] Saved to {path}")

    @classmethod
    def load(cls, path, env_obs_size: int, env_act_size: int, **_):
        path = Path(path).resolve()
        loaded = ocp.PyTreeCheckpointer().restore(path)
        hyper = loaded.get("hyperparams", {}) or {}
        agent = cls(
            env_obs_size=env_obs_size,
            env_action_size=env_act_size,
            action_low=jnp.asarray(loaded["action_low"]),
            action_high=jnp.asarray(loaded["action_high"]),
            scale=hyper.get("scale", 0.2),
            theta=hyper.get("theta", 0.15),
            dt=hyper.get("dt", 1e-2),
            clip=hyper.get("clip", 2.0),
            mu=hyper.get("mu", 0.0),
            seed=hyper.get("seed", 0),
        )
        print(f"OrnsteinUhlenbeck agent loaded from {path}")
        return agent


class Constant(Agent):
    """Agent producing a unique constant action."""

    def __init__(self, constant=0.0):
        self.constant = constant

    def initialize(self, observation_space, action_size, seed=None):
        self.action_size = action_size

    def step(self, observations, steps):
        return self._policy(observations)

    def test_step(self, observations, steps):
        return self._policy(observations)

    def _policy(self, observations):
        return np.full(self.action_size, self.constant)
