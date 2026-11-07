"""Replay buffer manager and transition schema handling."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from flashbax.buffers.flat_buffer import ExperiencePair, TransitionSample
from flashbax.buffers.trajectory_buffer import TrajectoryBufferSample

from roxie.utils.math import finite_or_zero

SCRUBBED_FIELDS = frozenset({"observation", "reward"})


def _rows(state, key, row_weights, batch_size, sequence_length):
    """flashbax's trajectory sample, with the row drawn by `row_weights`.

    Everything but the row draw is `flashbax.buffers.trajectory_buffer.sample`
    at `period=1`, which every buffer here uses; a flat buffer is that with a
    sequence of 2.
    """
    max_length = jax.tree.leaves(state.experience)[0].shape[1]
    max_time = jnp.where(state.is_full, max_length, state.current_index)
    head = jnp.where(state.is_full, state.current_index, 0)
    max_start = max_time - sequence_length
    num_valid = jnp.where(max_start >= 0, max_start + 1, 0)
    item_key, row_key = jax.random.split(key)
    start = (head + jax.random.randint(
        item_key, (batch_size,), 0, num_valid)) % max_length
    # Inverse CDF rather than `categorical`, which draws a Gumbel per row per
    # sample; unrolled, as the default search is a loop of kernels on a GPU.
    cdf = jnp.cumsum(row_weights)
    u = jax.random.uniform(row_key, (batch_size,), dtype=cdf.dtype) * cdf[-1]
    rows = jnp.minimum(
        jnp.searchsorted(cdf, u, side="right", method="scan_unrolled"),
        cdf.shape[0] - 1,
    )
    times = (start[:, None] + jnp.arange(sequence_length)) % max_length
    return jax.tree.map(lambda x: x[rows[:, None], times], state.experience)


@functools.partial(jax.jit, static_argnames=("replay",), donate_argnums=(0,))
def _scan_add(buffer_state, transitions, *, replay):
    """Adds a (T, B, ...) block of transitions via a scanned loop over time."""
    def add_one(bs, transition):
        return replay._store(bs, transition), None

    bs, _ = jax.lax.scan(add_one, buffer_state, transitions)
    return bs


class ReplayManager:
    """Manages buffer additions, layout adaptation, and transition scrubbing.

    Attributes:
        buffer: Underlying Flashbax buffer instance.
        prototype: Prototype transition structure used for schema verification.
        time_axis: Whether additions require an explicit leading time axis.
        task_ids: Each buffer row's task, for a multi-task env; None otherwise.
            A world keeps its task for the whole run, so a row holds one task's
            stream and weighting tasks is weighting rows.
    """

    def __init__(self, buffer, prototype, *, time_axis: bool = False,
                 task_ids=None, sample_batch_size: int | None = None,
                 sample_sequence_length: int | None = None):
        self.buffer = buffer
        self.prototype = prototype
        self.time_axis = bool(time_axis)
        self.task_ids = None if task_ids is None else np.asarray(task_ids)
        self._sample_batch_size = sample_batch_size
        self._sample_sequence_length = sample_sequence_length

        self.sample = buffer.sample
        self.can_sample = buffer.can_sample
        self._add_jit = None

    def draw(self, buffer_state, key, task_weights=None):
        """One off-policy batch, its rows weighted by task when weights are set.

        Args:
            buffer_state: the replay to sample.
            key: the draw's key.
            task_weights: `(K,)`, traced, need not sum to 1; None samples
              uniformly through flashbax itself.

        Returns:
            What `buffer.sample` returns, in the same layout.
        """
        if task_weights is None:
            return self.buffer.sample(buffer_state, key)
        counts = np.bincount(self.task_ids, minlength=task_weights.shape[0])
        per_row = (task_weights[self.task_ids]
                   / jnp.asarray(counts[self.task_ids], dtype=task_weights.dtype))
        sequence = _rows(
            buffer_state, key, per_row,
            self._sample_batch_size, self._sample_sequence_length,
        )
        if self.time_axis:
            return TrajectoryBufferSample(experience=sequence)
        return TransitionSample(experience=ExperiencePair(
            first=jax.tree.map(lambda x: x[:, 0], sequence),
            second=jax.tree.map(lambda x: x[:, 1], sequence),
        ))

    def init(self):
        """Initializes an empty buffer state matching the prototype schema."""
        return self.buffer.init(self.prototype)

    def add(self, buffer_state, transitions):
        """Adds one env-step batch of transitions (leaves shaped (B, ...)) from host."""
        if self._add_jit is None:
            self._add_jit = jax.jit(self.add_in_trace, donate_argnums=(0,))
        return self._add_jit(buffer_state, transitions)

    def add_in_trace(self, buffer_state, transitions):
        """Adds transitions directly within an active JAX trace."""
        return self._store(buffer_state, self.prepare(transitions))

    def add_block(self, buffer_state, transitions):
        """Adds a stacked (T, B, ...) block of transitions in a single scan dispatch."""
        return _scan_add(buffer_state, self.prepare(transitions), replay=self)

    def prepare(self, transition):
        """Filters unused fields and scrubs non-finite values from observations/rewards."""
        def field(name):
            value = getattr(transition, name)
            if value is None or getattr(self.prototype, name, None) is None:
                return None
            return finite_or_zero(value) if name in SCRUBBED_FIELDS else value

        return type(transition)(**{
            f.name: field(f.name) for f in dataclasses.fields(transition)
        })

    def _store(self, buffer_state, transition):
        """Dispatches an aligned transition to the Flashbax buffer."""
        if self.time_axis:
            transition = jax.tree.map(lambda x: x[:, None], transition)
        return self.buffer.add(buffer_state, transition)