"""Independent tasks presented to the training loop as one batched env.

Every task is its own env, built by an ordinary roxie builder from its own env
block, with its own physics, observation and reward. Nothing is shared between
tasks or detected about them; they only have to agree on an action space.

The composite gives each task a fixed, contiguous block of worlds, in one-hot
order, sized by the task weights at launch. It zero-pads every observation to
the widest task and appends the task's one-hot, and passes each task's own
reward and done flags through untouched. A world keeps its task for the whole
run: its physics belongs to its task's env, and moving it would mean recompiling.
Steering what the learner trains on is the replay's job instead
(`ReplayManager.draw`).

`MultiTaskEnv` steps the task envs side by side inside the trace, and looks
like any other env to the rollout, the agents and the replay.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from roxie.environment import functional
from roxie.environment.loader import EnvBundle, build_env
from roxie.environment.vector import BatchVectorEnv, Timestep, VecState


def allocate_worlds(weights, total: int) -> tuple[int, ...]:
    """Splits `total` worlds over tasks in proportion to `weights`.

    Largest-remainder rounding, so the counts sum to `total` exactly, and every
    task keeps at least one world.

    Raises:
        ValueError: on a non-positive weight, or fewer worlds than tasks.
    """
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w <= 0):
        raise ValueError(f"task weights {list(w)} must all be positive")
    if total < w.size:
        raise ValueError(f"{total} worlds cannot cover {w.size} tasks")
    raw = w / w.sum() * total
    counts = np.maximum(np.floor(raw).astype(int), 1)
    while counts.sum() < total:
        counts[np.argmax(raw - counts)] += 1
    while counts.sum() > total:
        counts[np.argmax(np.where(counts > 1, counts - raw, -np.inf))] -= 1
    return tuple(int(c) for c in counts)


class TaskLayout:
    """Where each task's worlds and observation columns sit in the batch.

    Attributes:
        names: The tasks, in one-hot order.
        counts: Each task's number of worlds, fixed for the run.
        widths: Each task's own observation width.
        width: The padded observation width, before the one-hot.
        slices: Each task's contiguous block of worlds.
        task_ids: Each world's task.
    """

    def __init__(self, names, counts, widths):
        self.names = tuple(names)
        self.counts = tuple(int(c) for c in counts)
        self.widths = tuple(int(w) for w in widths)
        self.width = max(self.widths)
        bounds = np.cumsum((0,) + self.counts)
        self.slices = tuple(
            slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])
        )
        self.task_ids = np.repeat(np.arange(len(self.names)), self.counts)
        self._eye = np.eye(len(self.names), dtype=np.float32)

    def compose(self, parts, xp):
        """Pads each task's `(n_k, D_k)` observations and appends the one-hot.

        `xp` is `jnp` or `np`.
        """
        rows = []
        for k, obs in enumerate(parts):
            n = obs.shape[0]
            pad = xp.zeros((n, self.width - obs.shape[1]), dtype=obs.dtype)
            hot = xp.broadcast_to(
                xp.asarray(self._eye[k], dtype=obs.dtype), (n, len(self.names))
            )
            rows.append(xp.concatenate([obs, pad, hot], axis=1))
        return xp.concatenate(rows, axis=0)


def _shared_action_space(envs):
    """The one action space every task agrees on."""
    first = envs[0].single_action_space
    for env in envs[1:]:
        space = env.single_action_space
        if (space.shape != first.shape
                or not np.array_equal(space.low, first.low)
                or not np.array_equal(space.high, first.high)):
            raise ValueError(
                f"tasks must share one action space; got {first} and {space}"
            )
    return first


def _task_metrics(names, steps, xp):
    """One value per task per step: its worlds' mean reward and own metrics.

    Shaped `(1,)` rather than per world, since a task's metric means nothing
    for another task's worlds; the epoch reduction means over the axis either
    way.
    """
    metrics = {}
    for name, step in zip(names, steps):
        metrics[f"task/{name}/reward"] = xp.mean(
            xp.asarray(step.reward, dtype=np.float32), keepdims=True
        )
        for key, value in step.info.get("metrics", {}).items():
            if key != "nonfinite":
                metrics[f"task/{name}/{key}"] = xp.mean(
                    xp.asarray(value, dtype=np.float32), keepdims=True
                )
    return metrics


class _TaskParams:
    """The per-task `params` hooks, over a tuple of each task's own params."""

    def __init__(self, envs, layout):
        self._funcs = tuple(env.func_env for env in envs)
        self._slices = layout.slices
        self.metadata = dict(getattr(self._funcs[0], "metadata", {}) or {})

    def init_params(self):
        params = tuple(f.init_params() for f in self._funcs)
        return None if all(p is None for p in params) else params

    def observe_params(self, params, info, terminated):
        if params is None:
            return None
        return tuple(
            f.observe_params(p, task_info, terminated[sl])
            for f, p, task_info, sl in zip(
                self._funcs, params, info["tasks"], self._slices)
        )

    def epoch_refresh(self, params):
        if params is None:
            return None, False
        out = [f.epoch_refresh(p) for f, p in zip(self._funcs, params)]
        return tuple(p for p, _ in out), any(inv for _, inv in out)


class MultiTaskEnv:
    """A driver running each task's own driver on its block of the batch.

    It sits beside `JaxVectorEnv` rather than on `FuncEnv`, which is one world:
    a world that could be any task would carry every task's state and, under
    `vmap`, run every task's physics. Whatever the backend, it only calls each
    task driver's `reset` and `step`.

    Attributes:
        envs: Each task's own vector env, in one-hot order.
        layout: Where each task sits in the batch.
        task_names: The tasks, in one-hot order.
        task_counts: Each task's number of worlds.
        task_ids: Each world's task.
        num_envs: The batch width, every task's worlds together.
    """

    def __init__(self, envs: dict):
        self.envs = tuple(envs.values())
        self.layout = TaskLayout(
            envs, [env.num_envs for env in self.envs],
            [functional.space_size(env.single_observation_space)
             for env in self.envs],
        )
        self.task_names = self.layout.names
        self.task_counts = self.layout.counts
        self.task_ids = self.layout.task_ids
        self.num_envs = int(sum(self.task_counts))
        self.single_action_space = _shared_action_space(self.envs)
        self.single_observation_space = functional.unbounded_box(
            self.layout.width + len(self.task_names)
        )
        # The eval loop's cap: no task's episode may be cut short by another's.
        self.max_episode_steps = max(env.max_episode_steps for env in self.envs)
        self.metadata = dict(self.envs[0].metadata)
        self.func_env = _TaskParams(self.envs, self.layout)

    def _params(self, params):
        return (None,) * len(self.envs) if params is None else params

    def reset(self, key, params=None, num_envs: int | None = None):
        """Fresh episodes for every task's worlds."""
        if num_envs is not None and int(num_envs) != self.num_envs:
            raise ValueError(
                f"a multi-task batch resets whole: {num_envs} != {self.num_envs}"
            )
        keys = jax.random.split(key, len(self.envs))
        states = tuple(
            env.reset(k, p)[0]
            for env, k, p in zip(self.envs, keys, self._params(params))
        )
        obs = self.layout.compose([state.obs for state in states], jnp)
        zeros = jnp.zeros(self.num_envs, dtype=jnp.float32)
        false = jnp.zeros(self.num_envs, dtype=jnp.bool_)
        return VecState(env_state=states, obs=obs), Timestep(
            obs=obs, reward=zeros, terminated=false, truncated=false, info={},
        )

    def step(self, state, action, key, reset_pool=None, params=None):
        """Steps every task on its own block of the batch."""
        keys = jax.random.split(key, len(self.envs))
        pools = (reset_pool.env_state if reset_pool is not None
                 else (None,) * len(self.envs))
        out = [
            env.step(sub, action[sl], k, pool, p)
            for env, sub, sl, k, pool, p in zip(
                self.envs, state.env_state, self.layout.slices, keys, pools,
                self._params(params))
        ]
        states = tuple(sub for sub, _ in out)
        steps = [step for _, step in out]
        metrics = {"nonfinite": jnp.concatenate(
            [step.info["metrics"]["nonfinite"] for step in steps])}
        metrics.update(_task_metrics(self.task_names, steps, jnp))
        timestep = Timestep(
            # Each task's driver already split pre- from post-auto-reset; the
            # buffer gets the former, acting the latter.
            obs=self.layout.compose([step.obs for step in steps], jnp),
            reward=jnp.concatenate([step.reward for step in steps]),
            terminated=jnp.concatenate([step.terminated for step in steps]),
            truncated=jnp.concatenate([step.truncated for step in steps]),
            info={"metrics": metrics,
                  "tasks": tuple(step.info for step in steps)},
        )
        obs = self.layout.compose([sub.obs for sub in states], jnp)
        return VecState(env_state=states, obs=obs), timestep


def _plain(cfg) -> Any:
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def build_multitask_env(
    tasks,
    *,
    task_defaults=None,
    weights=None,
    mode: str = "train",
    num_envs: int = 1,
    test_episodes: int = 1,
    seed: int = 0,
) -> EnvBundle:
    """Builds every task through its own builder, then composes them.

    Args:
        tasks: task name to that task's env block, merged over `task_defaults`.
        task_defaults: the keys every task shares, typically its builder and
          backend, as a backend group would set them.
        weights: task name to its relative share of the training worlds;
          equal shares when omitted.
        mode: injected; passed through to every task's builder.
        num_envs: injected; the training worlds, split by `weights`.
        test_episodes: injected; eval worlds PER TASK, whatever the weights.
        seed: offset per task, so no two tasks share a seed.

    Returns:
        The composed train and eval envs.

    Raises:
        ValueError: on weights for an unknown task, tasks on different
          backends, or tasks that disagree on the action space.
    """
    tasks = _plain(tasks)
    defaults = _plain(task_defaults) or {}
    weights = _plain(weights) or {}
    unknown = sorted(set(weights) - set(tasks))
    if unknown:
        raise ValueError(f"weights for unknown task(s) {unknown}")
    names = tuple(tasks)
    counts = allocate_worlds([weights.get(n, 1.0) for n in names], num_envs)

    train, test = {}, {}
    for k, name in enumerate(names):
        block = {**defaults, **(tasks[name] or {})}
        block.setdefault("seed", int(seed) + 2 * k)
        bundle = build_env(
            block, mode=mode, num_envs=counts[k], test_episodes=test_episodes,
        )
        train[name], test[name] = bundle.env, bundle.test_env

    batched = [isinstance(env, BatchVectorEnv) for env in train.values()]
    if any(batched) and not all(batched):
        raise ValueError("every task must run on the same backend")
    return EnvBundle(
        env=MultiTaskEnv(train), test_env=MultiTaskEnv(test), env_cfg=None,
    )
