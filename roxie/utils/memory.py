"""Replay buffer manager and transition schema handling."""

import dataclasses
import functools

import jax

from roxie.utils.math import finite_or_zero

SCRUBBED_FIELDS = frozenset({"observation", "reward"})


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
    """

    def __init__(self, buffer, prototype, *, time_axis: bool = False):
        self.buffer = buffer
        self.prototype = prototype
        self.time_axis = bool(time_axis)

        self.sample = buffer.sample
        self.can_sample = buffer.can_sample
        self._add_jit = None

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