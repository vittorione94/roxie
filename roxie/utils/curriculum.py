"""Task curricula: how often the learner samples each task of a multi-task env.

A curriculum reads each task's training score once per epoch and returns the
weights the replay draws tasks by (`ReplayManager.draw`). It never touches an
env, so it is the same on every backend, and moving its weights costs no
recompile: they reach the learning pass as a traced array.

What it does not move is which tasks the envs collect: each task's share of
worlds is fixed at launch by the env block's `weights`. A curriculum decides
what the learner trains on, from whatever the worlds collected.

The trainer drives it: `start` once with the task names, then `update` at every
epoch boundary.
"""

import numpy as np


class WeightedTasks:
    """Fixed sampling weights by task name; a curriculum that never moves.

    Attributes:
        weights: task name to its relative weight; a task left out weighs 1.
    """

    def __init__(self, weights):
        self.weights = {str(k): float(v) for k, v in dict(weights).items()}
        self._weights = None

    def start(self, task_names):
        """Orders the weights by the env's tasks."""
        unknown = sorted(set(self.weights) - set(task_names))
        if unknown:
            raise ValueError(f"weights for unknown task(s) {unknown}")
        w = np.asarray([self.weights.get(n, 1.0) for n in task_names])
        if np.any(w <= 0):
            raise ValueError(f"task weights {self.weights} must be positive")
        self._weights = w / w.sum()

    def update(self, scores):
        """Returns the fixed weights, whatever `scores` says."""
        del scores
        return self._weights


class LearningProgressTasks:
    """Weights tasks by how fast their score is changing.

    Progress is the gap between a fast and a slow moving average of each task's
    score, so an improving or collapsing task is sampled more and a solved or
    stalled one less. `floor` is spread uniformly, so no task starves.

    Attributes:
        score_scale: what scores are divided by before averaging; a dm_control
            episode pays at most 1 per step, so 1000 for its 1000-step episodes.
        floor: the share of every draw spread uniformly over the tasks.
        fast_rate: step size of the fast moving average.
        slow_rate: step size of the slow moving average.
    """

    def __init__(self, score_scale, floor=0.2, fast_rate=0.5, slow_rate=0.1):
        self.score_scale = float(score_scale)
        self.floor = float(floor)
        self.fast_rate = float(fast_rate)
        self.slow_rate = float(slow_rate)
        self._fast = self._slow = None

    def start(self, task_names):
        """Sizes the averages for the env's tasks."""
        self._fast = np.full(len(task_names), np.nan)
        self._slow = np.full(len(task_names), np.nan)

    def update(self, scores):
        """Folds an epoch's per-task scores in and returns the new weights.

        Args:
            scores: one score per task; `nan` for a task that finished no
              episode this epoch, which keeps its averages where they were.
        """
        s = np.asarray(scores, dtype=np.float64) / self.score_scale
        seen = ~np.isnan(s)
        first = seen & np.isnan(self._fast)
        self._fast[first] = self._slow[first] = s[first]
        later = seen & ~first
        self._fast[later] += self.fast_rate * (s[later] - self._fast[later])
        self._slow[later] += self.slow_rate * (s[later] - self._slow[later])
        # A task with no score yet counts as maximally interesting.
        progress = np.where(np.isnan(self._fast), 1.0,
                            np.abs(self._fast - self._slow))
        k = progress.size
        total = progress.sum()
        focus = progress / total if total > 0 else np.full(k, 1.0 / k)
        return self.floor / k + (1.0 - self.floor) * focus
