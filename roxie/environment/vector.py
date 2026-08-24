"""Driving a batched environment the trainer can jit, scan and step.

Two implementations of one surface:

    single_observation_space / single_action_space / num_envs / metadata
    reset(key, params=None, num_envs=None)          -> (VecState, Timestep)
    step(state, action, key, reset_pool, params)    -> (VecState, Timestep)

``JaxVectorEnv`` vmaps a ``roxie.environment.functional.FuncEnv`` across worlds;
``EnvPoolVectorEnv`` fronts a C++ pool that batches natively. Everything above
them — trainer, learner, agents — sees only this surface and never asks which it
is driving.

WHY NOT ``gymnasium.envs.functional_jax_env.FunctionalJaxVectorEnv``
--------------------------------------------------------------------
The shape below is deliberately Gymnasium's; the implementation cannot be.
Upstream's vector env is unusable here for three independent reasons, all
visible in its source:

  1. ``step`` branches on ``if jnp.any(self.prev_done):`` — a device-to-host
     sync EVERY step. A release cell runs 5e7 env steps; that one line
     would dominate the loop.
  2. It resets with ``self.state.at[to_reset].set(...)``, which assumes the
     state is a single array. Roxie's states are pytrees (``mjx.Data``), and
     ``.at[]`` does not exist on a pytree — it raises on every env in this repo.
  3. It keeps the state on ``self``, so warmup cannot be a ``lax.scan``. Roxie
     fills the replay buffer in ONE dispatch; a per-step Python loop there costs
     minutes at the step counts warmup needs.

Hence: same public shape, functional core. The caller carries the state, and a
whole step — physics, reward, termination, auto-reset — is one jittable
function.

AUTO-RESET
----------
Roxie resets a done env IN THE SAME STEP, by gathering a fresh start from a
pre-built pool of reset states, rather than on the next step the way
Gymnasium's ``AutoresetMode.NEXT_STEP`` does. The reason is mechanical: a real
reset of "however many envs happen to be done" has a data-dependent shape and
cannot be jitted, whereas a gather from a fixed-size pool can. The pool is
passed IN to ``step`` rather than read off ``self`` so that regenerating it each
epoch does not invalidate the caller's compiled step.

This is why ``step`` returns two things. ``Timestep`` holds the PRE-reset
values — the true next observation, which is what the replay buffer must
store — while the returned ``VecState`` holds the POST-reset observation, which
is what the next action is selected from. For a pool that resets in C++ the two
observations are the same array; see ``EnvPoolVectorEnv``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from roxie.environment import functional


@struct.dataclass
class VecState:
    """The carry: everything the next step needs, and nothing else.

    ``obs`` rides in the state rather than being recomputed because the
    auto-reset gather has to touch it — an env that just reset must act from its
    NEW observation, not the terminal one.
    """

    env_state: Any
    obs: jax.Array
    # Per-env elapsed control steps, for the driver's own time limit. ``None``
    # for pools, which count internally.
    steps: Any = None


class Timestep(NamedTuple):
    """Gymnasium's 5-tuple, pre-auto-reset. Unpacks as
    ``obs, reward, terminated, truncated, info``."""

    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: dict


class JaxVectorEnv:
    """A ``FuncEnv`` batched across ``num_envs`` worlds.

    Args:
        func_env: the environment. NOT mutated — unlike Gymnasium, which calls
            ``func_env.transform(jax.vmap)`` in place, this builds its vmapped
            callables locally. A builder may hand the SAME env object to the
            train and the eval driver — that is the point of doing so, since the
            mjx model and any per-env reference data are then loaded once — and
            mutating it in place would double-vmap the second driver. It also
            keeps the env callable on single states, which ``play.py`` needs.
        num_envs: worlds stepped per call.
        max_episode_steps: the driver's step limit; 0 disables it. Reported as
            TRUNCATION, never termination.
    """

    def __init__(self, func_env, num_envs: int, max_episode_steps: int = 0):
        self.func_env = func_env
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)

        self.single_observation_space = func_env.observation_space
        self.single_action_space = func_env.action_space
        self.metadata = dict(getattr(func_env, "metadata", {}) or {})

        # ``params`` broadcasts (in_axes None) so it stays a traced argument the
        # caller can vary without recompiling — the whole point of ``params``.
        self._v_initial = jax.vmap(func_env.initial, in_axes=(0, None))
        self._v_transition = jax.vmap(func_env.transition, in_axes=(0, 0, 0, None))
        self._v_observation = jax.vmap(func_env.observation, in_axes=(0, 0, None))
        self._v_reward = jax.vmap(func_env.reward, in_axes=(0, 0, 0, 0, None))
        self._v_terminal = jax.vmap(func_env.terminal, in_axes=(0, 0, None))
        self._v_truncal = jax.vmap(func_env.truncal, in_axes=(0, 0, None))
        self._v_state_info = jax.vmap(func_env.state_info, in_axes=(0, None))
        self._v_transition_info = jax.vmap(
            func_env.transition_info, in_axes=(0, 0, 0, None)
        )

    # -- construction --------------------------------------------------------

    def reset(self, key, params=None, num_envs: int | None = None):
        """Fresh episodes for ``num_envs`` worlds (default: this env's count).

        The count is an argument because the auto-reset pool is built with this
        same call and need not be the same size as the batch being stepped.
        """
        n = self.num_envs if num_envs is None else int(num_envs)
        keys = jax.random.split(key, n)
        env_state = self._v_initial(keys, params)
        obs = self._v_observation(env_state, keys, params)
        state = VecState(
            env_state=env_state, obs=obs, steps=jnp.zeros(n, dtype=jnp.int32),
        )
        zeros = jnp.zeros(n, dtype=jnp.float32)
        false = jnp.zeros(n, dtype=jnp.bool_)
        return state, Timestep(
            obs=obs, reward=zeros, terminated=false, truncated=false,
            info=self._v_state_info(env_state, params),
        )

    # -- the step ------------------------------------------------------------

    def step(
        self, state: VecState, action, key, reset_pool: VecState | None = None,
        params=None,
    ):
        """One control step for every world, then the auto-reset gather.

        Pure and jittable end to end: the caller owns ``state``, ``reset_pool``
        is an argument (see the module docstring), and ``params`` is traced.

        ``reset_pool=None`` disables the auto-reset. That is what the evaluation
        rollout wants: it runs each episode to its own end and masks finished
        worlds, so resetting them would step episodes nobody scores.
        """
        step_key, pool_key = jax.random.split(key)
        keys = jax.random.split(step_key, self.num_envs)

        prev_env_state = state.env_state
        next_env_state = self._v_transition(prev_env_state, action, keys, params)
        reward = self._v_reward(prev_env_state, action, next_env_state, keys, params)
        obs = self._v_observation(next_env_state, keys, params)
        info = self._v_transition_info(
            prev_env_state, action, next_env_state, params,
        )

        # The three-way split. `terminal` is failure, `truncal` is the env's own
        # non-failure cutoff (a reference clip running out), and the step limit
        # is the driver's. An env-internal truncation CLEARS termination —
        # otherwise a value-based agent zeroes its bootstrap at the cutoff and Q
        # collapses there — while the step limit does not, since a genuine fall
        # on the very last step is still a fall.
        terminal = self._v_terminal(next_env_state, keys, params)
        truncal = self._v_truncal(next_env_state, keys, params)
        steps = state.steps + 1
        step_truncated = (
            steps >= self.max_episode_steps if self.max_episode_steps > 0
            else jnp.zeros_like(terminal)
        )
        terminated = jnp.logical_and(terminal, jnp.logical_not(truncal))
        truncated = jnp.logical_or(truncal, step_truncated)
        done = jnp.logical_or(jnp.logical_or(terminal, truncal), step_truncated)

        timestep = Timestep(
            obs=obs, reward=reward, terminated=terminated, truncated=truncated,
            info=info,
        )
        stepped = VecState(env_state=next_env_state, obs=obs, steps=steps)
        if reset_pool is None:
            return stepped, timestep
        return self._autoreset(stepped, done, reset_pool, pool_key), timestep

    def _autoreset(self, stepped: VecState, done, reset_pool: VecState, key):
        """Replace every done world with a random draw from the reset pool.

        A gather rather than a real reset, because the number of done envs is
        data-dependent and a real reset therefore cannot be jitted. The pool is
        regenerated each epoch by the caller, so the start distribution stays
        fresh — and tracks whatever the env did to its own ``params`` at the
        epoch boundary.
        """
        # The pool is gathered FROM, so its size need not match ``num_envs``;
        # read it off the pool rather than assuming they are equal.
        pool_size = reset_pool.obs.shape[0]
        idx = jax.random.randint(key, (self.num_envs,), 0, pool_size)

        def _leaf(pool_leaf, s):
            # Leaves with no per-env leading dim — warp's world-flattened global
            # contact arena is the real case — would be indexed out of bounds by
            # the gather. The physics recomputes them every step, so keeping the
            # stepped value is both correct and what the pre-refactor loop did.
            if not (isinstance(s, jnp.ndarray) and s.shape[:1] == done.shape):
                return s
            return jnp.where(
                done.reshape(done.shape + (1,) * (s.ndim - 1)), pool_leaf[idx], s,
            )

        return jax.tree.map(_leaf, reset_pool, stepped)


def _dict_obs_keys(space: Any) -> tuple[str, ...] | None:
    """The observation keys, in order, when a pool's obs space is a ``Dict``.

    ``None`` for the flat case. EnvPool's gym-MuJoCo tasks declare a single Box;
    its dm_control tasks declare dm_control's own OrderedDict of named
    observation groups (``position``, ``velocity``, ``touch``, ...), which is
    what a policy has to be handed as one vector. The ORDER is the space's, i.e.
    dm_control's declaration order — not sorted — because that is the order
    dm_control itself flattens in, and a policy trained against one ordering
    cannot be evaluated against another.
    """
    spaces = getattr(space, "spaces", None)
    if not isinstance(spaces, dict):
        return None
    return tuple(spaces)


def _as_box(space: Any, unbounded: bool = False) -> Any:
    """Normalize a pool's space object to a ``gymnasium.spaces.Box``.

    Pools hand back a real Box (EnvPool), a ``Dict`` of Boxes (EnvPool's
    dm_control tasks — flattened here to the concatenated vector the pool's
    observations are flattened to), or a minimal stand-in carrying only
    ``shape``/``low``/``high``. ``unbounded`` supplies +-inf bounds for a
    stand-in that declares none, which is the honest space for an unclipped
    MuJoCo observation vector.
    """
    keys = _dict_obs_keys(space)
    if keys is not None:
        size = sum(int(np.prod(space.spaces[k].shape)) for k in keys)
        return functional.unbounded_box(size)
    low, high = getattr(space, "low", None), getattr(space, "high", None)
    if low is None or high is None:
        if not unbounded:
            raise ValueError(f"space {space!r} declares no low/high bounds")
        return functional.unbounded_box(int(np.prod(space.shape)))
    return functional.box(low, high, shape=tuple(space.shape))


class EnvPoolVectorEnv:
    """A C++ pool (EnvPool, or a hand-written one) behind the same surface.

    The pool already batches, auto-resets and time-limits internally and already
    returns ``(obs, reward, terminated, truncated, info)`` — this class is
    packaging, not translation.

    ``VecState.env_state`` is ``None`` and ``VecState.obs`` is the same array as
    ``Timestep.obs``: the pool resets in C++ and hands back only the POST-reset
    observation, so the terminal transition's stored ``next_obs`` is the reset
    obs. That is a pre-existing property of this path, not something introduced
    here, and it is harmless because a true termination zeroes the bootstrap
    anyway.
    """

    # Read by the startup banner in place of a physics ``impl``: a pool has no
    # MJX backend, it IS the backend.
    metadata = {"jax": False, "impl": "envpool"}

    # Optional pool hook, forwarded verbatim so it reaches the rollout through
    # this class. ``epoch_refresh()`` is the pool's counterpart to
    # ``FuncEnv.epoch_refresh``: whatever it regenerates once per epoch — its
    # own reset pool, a start distribution it adapts — it does in there. There
    # is no ``params`` to thread, because a C++ pool steps in plain Python and
    # can just own the mutable state.
    _POOL_HOOKS = ("epoch_refresh",)

    def __init__(
        self, pool: Any, num_envs: int | None = None,
        max_episode_steps: int = 1000, rebuild=None,
    ):
        self.max_episode_steps = int(max_episode_steps)
        # ``rebuild(seed) -> pool`` when the builder can reconstruct this pool;
        # see ``reseed``. None keeps whatever reset determinism the pool has.
        self._rebuild = rebuild
        self._num_envs = num_envs
        self._bind_pool(pool)

        # Both pool flavours expose the SINGLE-env spaces (EnvPool's gymnasium
        # API does), so they are the single_* spaces directly.
        self.single_observation_space = _as_box(
            pool.observation_space, unbounded=True,
        )
        self.single_action_space = _as_box(pool.action_space)
        # Non-None when this pool reports observations as named groups rather
        # than one vector; see `_flat_obs`.
        self._obs_keys = _dict_obs_keys(pool.observation_space)

    def _bind_pool(self, pool: Any) -> None:
        self._pool = pool
        # EnvPool exposes ``num_envs``; a pool that doesn't gets it from the
        # builder, and failing both it is fixed up by the first ``reset``.
        self.num_envs = int(
            getattr(pool, "num_envs", None) or self._num_envs or 0
        ) or None
        for name in self._POOL_HOOKS:
            attr = getattr(pool, name, None)
            if attr is not None:
                setattr(self, name, attr)

    def reseed(self, seed: int) -> bool:
        """Rebuild the pool at ``seed`` so the next ``reset()`` is reproducible.

        EnvPool's own ``seed()`` is a documented no-op as of 1.2.5 and
        ``reset()`` advances the pool's RNG, so a pool reset once per epoch for
        evaluation starts from different states every time — a change in
        ``test/score`` could then be a different draw rather than a better
        policy. The JAX eval path pins its reset keys for exactly this reason;
        rebuilding (~3ms for a 5-env pool) is how a C++ pool gets the same
        guarantee. Returns whether it could.
        """
        if self._rebuild is None:
            return False
        self._bind_pool(self._rebuild(seed))
        return True

    def _flat_obs(self, obs: Any) -> np.ndarray:
        """The pool's observation as one ``(num_envs, obs_dim)`` float32 array.

        A dm_control pool hands back a mapping of named groups; concatenating
        them in the space's declared order is what dm_control's own
        ``flatten_observation`` does, and it is the only step in this class that
        is more than a dtype cast. Leaves are reshaped rather than assumed
        2-D — a scalar group like walker's ``height`` arrives as ``(n,)``.
        """
        if self._obs_keys is None:
            return np.asarray(obs, dtype=np.float32)
        parts = []
        for key in self._obs_keys:
            leaf = obs[key] if hasattr(obs, "__getitem__") else getattr(obs, key)
            leaf = np.asarray(leaf, dtype=np.float32)
            parts.append(leaf.reshape(leaf.shape[0], -1))
        return np.concatenate(parts, axis=1)

    def reset(self, key=None, params=None, num_envs: int | None = None):
        del key, params, num_envs  # the pool owns its own RNG and batch size
        obs, info = self._pool.reset()
        obs = self._flat_obs(obs)
        n = obs.shape[0]
        self.num_envs = n
        false = np.zeros(n, dtype=bool)
        return VecState(env_state=None, obs=obs), Timestep(
            obs=obs, reward=np.zeros(n, dtype=np.float32),
            terminated=false, truncated=false,
            info=info if isinstance(info, dict) else {},
        )

    def step(self, state: VecState, action, key=None, reset_pool=None, params=None):
        del state, key, reset_pool, params
        obs, reward, terminated, truncated, info = self._pool.step(
            np.asarray(action)
        )
        obs = self._flat_obs(obs)
        timestep = Timestep(
            obs=obs,
            reward=np.asarray(reward, dtype=np.float32),
            terminated=np.asarray(terminated, dtype=bool),
            truncated=np.asarray(truncated, dtype=bool),
            info=info if isinstance(info, dict) else {},
        )
        # Same array in both slots — see the class docstring.
        return VecState(env_state=None, obs=obs), timestep
