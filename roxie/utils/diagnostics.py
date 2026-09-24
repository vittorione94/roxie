"""Epoch health metric accumulation and reduction rules."""

from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

import jax.numpy as jnp

from roxie.utils.precision import FLOAT

# Metric keys reduced using max rather than a weighted average.
DIAGNOSTIC_MAX_KEYS = frozenset({"pre_act_max"})

# Metric keys summed across an epoch rather than averaged.
DIAGNOSTIC_SUM_KEYS = frozenset({"kl_early_stops"})


def reduce_learning_passes(
    passes: Iterable[Tuple[int, Dict[str, Any]]],
    *,
    max_keys: FrozenSet[str] = DIAGNOSTIC_MAX_KEYS,
    sum_keys: FrozenSet[str] = DIAGNOSTIC_SUM_KEYS,
) -> Dict[str, float]:
    """Collapses recorded learning-pass diagnostics into scalar values per key.

    Args:
        passes: Iterable of (steps, diagnostics_dict) pairs recorded during the epoch.
        max_keys: Metric names to reduce via maximum.
        sum_keys: Metric names to reduce via sum.

    Returns:
        A dictionary mapping metric keys to host float values.
    """
    passes = list(passes)
    if not passes:
        return {}

    weights = jnp.stack(
        [jnp.asarray(steps, jnp.float32) for steps, _ in passes]
    )
    out = {}
    for key in passes[0][1]:
        values = jnp.stack(
            [jnp.asarray(record[key], FLOAT) for _, record in passes]
        )
        if key in max_keys:
            reduced = jnp.max(values)
        elif key in sum_keys:
            reduced = jnp.sum(values)
        else:
            reduced = jnp.sum(values * weights) / jnp.sum(weights)
        out[key] = float(reduced)
    return out


class DiagnosticsTracker:
    """Buffers, reduces, and namespaces epoch diagnostics for a single component.

    Attributes:
        prefix: Namespace prefix applied to output keys (e.g., 'td3').
        rate_key: Metric key for tracking total work rate; None to disable.
        max_keys: Keys reduced using max.
        sum_keys: Keys reduced using sum.
    """

    def __init__(
        self,
        prefix: str,
        *,
        rate_key: Optional[str] = "updates_per_env_step",
        max_keys: FrozenSet[str] = DIAGNOSTIC_MAX_KEYS,
        sum_keys: FrozenSet[str] = DIAGNOSTIC_SUM_KEYS,
    ):
        self.prefix = prefix
        self.rate_key = rate_key
        self.max_keys = max_keys
        self.sum_keys = sum_keys

        self._learning_passes: List[Tuple[int, Dict[str, Any]]] = []
        self.total_steps = 0

    def key(self, name: str) -> str:
        """Formats a metric name under this tracker's namespace prefix."""
        return f"{self.prefix}/{name}"

    @property
    def pending_learning_passes(self) -> int:
        """Number of learning passes recorded since the last drain."""
        return len(self._learning_passes)

    @property
    def pending_steps(self) -> Any:
        """Total steps covered by pending learning passes."""
        return sum(steps for steps, _ in self._learning_passes)

    def record(self, diagnostics: Dict[str, Any], steps: Any) -> None:
        """Buffers one learning pass's diagnostic scalars.

        Args:
            diagnostics: Dictionary mapping metric names to device scalars.
            steps: Step count covered by this pass (host int or device scalar).
        """
        self._learning_passes.append((steps, diagnostics))
        self.total_steps = self.total_steps + steps

    def pop(self, denom: int = 0) -> Dict[str, float]:
        """Reduces and resets buffered metrics for the epoch.

        Args:
            denom: External step count (e.g., total env steps) to compute the update rate.

        Returns:
            Namespaced dictionary of reduced metric values, or {} if empty.
        """
        passes, self._learning_passes = self._learning_passes, []
        if not passes:
            return {}

        out = {
            self.key(key): value
            for key, value in reduce_learning_passes(
                passes, max_keys=self.max_keys, sum_keys=self.sum_keys
            ).items()
        }
        if self.rate_key and denom > 0:
            out[self.key(self.rate_key)] = float(self.total_steps) / denom
        return out