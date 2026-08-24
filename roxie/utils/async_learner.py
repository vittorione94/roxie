"""Asynchronous actor-learner for the CPU-physics / GPU-learner split.

The envpool training loop is otherwise fully serial: it steps the CPU physics
pool, then runs the GPU gradient burst, with each device idle while the other
works. For agents where the learner is the dominant cost (e.g. D4PG's
distributional critic), that serialization caps throughput well below the CPU
physics ceiling — the GPU sits idle through every physics wave and vice versa.

``AsyncLearner`` moves the gradient updates onto a background thread that is the
*sole owner* of ``agent.state`` (networks, optimizers, replay buffer, obs
stats). The main (acting) thread never touches ``agent.state``: it selects
actions from a **behaviour actor** snapshot the learner publishes, steps the
envs on CPU, and hands the resulting transitions to the learner through a queue.
So the CPU physics and the GPU gradient bursts overlap in wall-clock time.

Why this ownership split matters: the agents' fused ``_grad_steps`` donates the
whole train state (replay buffer included) to avoid copying it. That donation is
only safe if nothing else references the buffer concurrently — which holds
precisely because the learner thread is the *only* thing that touches
``agent.state`` while it runs. The acting thread reads from an independent
behaviour actor and pushes raw transition arrays; it never aliases the buffer.

Semantics vs. the synchronous loop:
  * Replay ratio is preserved: the learner runs one ``learning_steps`` burst
    each time the number of *buffered* transitions crosses a
    ``steps_between_updates`` boundary past ``steps_before_learning`` — the same
    cadence the sync ``update`` uses, just off the main thread. If the GPU can't
    keep up it runs bursts back-to-back to catch up; if it's ahead it waits for
    data. So the grad-steps-per-env-step ratio matches sync on average and can
    never run ahead of collected data.
  * The behaviour policy lags the learner by up to one burst (standard for
    async off-policy RL); transitions collected under a slightly stale policy
    are fine for an off-policy replay buffer.

Concurrency contract: the only cross-thread shared objects are the transition
queue (thread-safe) and a lock-guarded snapshot/metrics slot. The lock is held
only for cheap reference swaps, never across device compute, so the GIL-free
JAX dispatch on both threads can actually overlap. It is NOT safe to call
``agent.step`` / ``agent.save`` (which read ``agent.state``) while the learner
runs; the trainer brackets eval and checkpointing with ``pause()`` / ``resume()``
so those observe a consistent, quiescent state.
"""

from __future__ import annotations

import queue
import threading

import jax
import jax.numpy as jnp
from flax import nnx


class AsyncLearner:
    def __init__(self, agent, agent_key, *, initial_steps: int = 0,
                 chunk: int = 8, queue_maxsize: int = 512):
        self._agent = agent
        self._key = agent_key
        self.learning_steps = int(agent.learning_steps)
        self.steps_before_learning = int(agent.steps_before_learning)
        self.steps_between_updates = int(agent.steps_between_updates)
        # `chunk` is how many fused grad steps the learner submits at a time —
        # small so the GPU stream keeps freeing up for the acting thread's forward
        # pass, which is what lets CPU physics overlap GPU learning. `_ratio` is
        # the sync loop's grad-steps-per-env-step, reproduced on average below.
        self._chunk = max(1, min(int(chunk), self.learning_steps))
        self._ratio = self.learning_steps / max(1, self.steps_between_updates)

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._lock = threading.Lock()

        # Published behaviour snapshot (learner -> main), version-stamped so the
        # acting thread only re-syncs its behaviour actor when it changes.
        self._snapshot = None            # (actor_param_state, obs_stats)
        self._snapshot_version = 0

        # Learner-owned counters/metrics, read by the main thread for logging.
        # `_added_steps` seeds from transitions already buffered synchronously
        # (the warmup fill) so the first burst fires on the same total-step
        # boundary the sync loop would use.
        self._added_steps = int(initial_steps)
        self._grad_steps = 0
        self._losses: list = []          # (actor_loss, critic_loss) since drain

        # Control flags.
        self._stop = threading.Event()
        self._pause_req = threading.Event()
        self._paused = threading.Event()   # learner acks it is idle/paused
        self._resume = threading.Event()
        self._resume.set()

        self._thread = threading.Thread(
            target=self._loop, name="async-learner", daemon=True
        )

        # Behaviour-side state, owned by the ACTING thread (see `act`). Held here
        # rather than in the training loop so the snapshot-versioning protocol
        # stays this class's business — the loop just calls `act`.
        self._behavior_actor = None
        self._behavior_stats = None
        self._behavior_version = -1

    # -- construction -------------------------------------------------------

    @classmethod
    def started(cls, agent, agent_key, state, *, initial_steps=0, chunk=8):
        """Build, warm the two programs the two threads will race on, and start.

        Both compiles happen HERE, on the calling thread, before the learner
        exists: the acting thread's behaviour path and the learner thread's
        fixed-size chunk burst are different programs from anything compiled so
        far, and a cold compile of either mid-run would stall the other.
        """
        import copy

        agent_key, learner_key, chunk_key = jax.random.split(agent_key, 3)
        learner = cls(agent, learner_key, initial_steps=initial_steps, chunk=chunk)

        learner._behavior_actor = copy.deepcopy(agent.state.actor)
        learner._behavior_stats = agent.state.obs_stats
        warm_action, _ = agent.select_action(
            learner._behavior_actor, learner._behavior_stats,
            state.obs, agent_key, evaluate=False,
        )
        jax.block_until_ready(warm_action)

        agent.learn(chunk_key, n_steps=learner._chunk)
        jax.block_until_ready(jax.tree.leaves(nnx.state(agent.state)))

        learner.start()
        print("Async learner started "
              "(CPU acting overlaps GPU gradient bursts).", flush=True)
        return learner

    # -- main-thread API ----------------------------------------------------

    def start(self):
        # Seed an initial behaviour snapshot so the acting thread can act from
        # step 0 (pre-learning behaviour = the current, warmup-era actor).
        self._publish_snapshot()
        self._thread.start()

    def act(self, obs, key):
        """Select actions from the behaviour snapshot — decoupled from the
        learner's live, mutating networks — re-syncing only when it advances."""
        version, snapshot = self._latest_snapshot()
        if version != self._behavior_version and snapshot is not None:
            nnx.update(self._behavior_actor, snapshot[0])
            self._behavior_stats = snapshot[1]
            self._behavior_version = version
        return self._agent.select_action(
            self._behavior_actor, self._behavior_stats, obs, key, evaluate=False,
        )

    def observe(self, prev_obs, timestep, actions, steps):
        """Hand the transition to the learner thread. The action travels WITH it
        rather than being read off `agent.last_action` — `select_action` never
        sets that, and the acting thread would overwrite it anyway. `steps` is
        unused: the learner paces itself off the transitions it has actually
        buffered, not the acting thread's count.
        """
        del steps
        self.push(
            prev_obs,
            actions,
            timestep.reward,
            timestep.terminated,
            timestep.truncated,
            timestep.obs,
        )

    def push(self, prev_obs, action, reward, termination, truncation, next_obs):
        """Hand one env-step transition batch to the learner. Blocks only if the
        learner has fallen far behind (bounded queue = natural backpressure)."""
        self._queue.put(
            (prev_obs, action, reward, termination, truncation, next_obs)
        )

    def _latest_snapshot(self):
        """`(version, (actor_params, obs_stats))`; snapshot is None before start."""
        with self._lock:
            return self._snapshot_version, self._snapshot

    def drain(self):
        """Return `(grad_steps_since_start, [(actor_loss, critic_loss), ...])` and
        clear the loss buffer. Losses are device arrays; the caller reduces them
        host-side at epoch boundaries exactly like the sync path."""
        with self._lock:
            losses = self._losses
            self._losses = []
            return self._grad_steps, losses

    def pause(self):
        """Block until the learner is quiescent so the caller can safely read
        ``agent.state`` (eval / checkpoint). A burst in flight finishes first."""
        self._resume.clear()
        self._pause_req.set()
        self._paused.wait()

    def resume(self):
        self._pause_req.clear()
        self._paused.clear()
        self._resume.set()

    def stop(self):
        self._stop.set()
        self._resume.set()
        # Unblock a possible blocking queue.get in the learner loop.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=30)

    # -- learner thread -----------------------------------------------------

    def _add_item(self, item) -> int:
        if item is None:  # stop / wake sentinel
            return 0
        prev_obs, action, reward, termination, truncation, next_obs = item
        self._agent.add_transitions(
            prev_obs, action, reward, termination, truncation, next_obs
        )
        return int(reward.shape[0])

    def _drain_queue(self) -> int:
        """Add every currently-queued transition batch to the buffer (the
        learner is the sole owner of the buffer)."""
        added = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            added += self._add_item(item)
        if added:
            with self._lock:
                self._added_steps += added
        return added

    def _publish_snapshot(self):
        # Independent on-device COPIES of the learner's current actor params and
        # obs stats. A plain reference would alias the buffers the very next
        # `learn()` donates (frees), leaving the acting thread reading a freed
        # buffer. The copy op is issued before that donation on the same GPU
        # stream, so it survives. Kept on device so the step path stays jnp.
        params = jax.tree.map(jnp.copy, nnx.state(self._agent.state.actor, nnx.Param))
        obs_stats = jax.tree.map(jnp.copy, self._agent.state.obs_stats)
        with self._lock:
            self._snapshot = (params, obs_stats)
            self._snapshot_version += 1

    def _loop(self):
        while not self._stop.is_set():
            if self._pause_req.is_set():
                self._drain_queue()   # don't strand buffered data across a pause
                self._paused.set()
                self._resume.wait()
                continue

            self._drain_queue()

            with self._lock:
                added = self._added_steps
                grads = self._grad_steps

            # Target cumulative grad steps for the data collected so far: one full
            # `learning_steps` worth at the `steps_before_learning` boundary (as
            # the sync loop does), then `ratio` more per env-step. Submitting the
            # deficit in `chunk`-sized pieces preserves the replay ratio on
            # average while keeping each GPU submission short.
            if added < self.steps_before_learning:
                target = 0.0
            else:
                target = self.learning_steps + self._ratio * (
                    added - self.steps_before_learning
                )

            if grads >= target:
                # Ahead of (or level with) the schedule — wait for more data
                # rather than busy-spin, so the acting thread's cores aren't
                # starved by a spinning learner.
                try:
                    item = self._queue.get(timeout=0.005)
                except queue.Empty:
                    continue
                n = self._add_item(item)
                if n:
                    with self._lock:
                        self._added_steps += n
                continue

            # Submit one fixed-size chunk. Keeping the size constant means a
            # single compiled `_grad_steps` program (n_steps is static); the
            # at-most-`chunk` overshoot past the target is negligible for the
            # replay ratio.
            self._key, burst_key = jax.random.split(self._key)
            actor_loss, critic_loss = self._agent.learn(
                burst_key, n_steps=self._chunk
            )
            with self._lock:
                self._grad_steps += self._chunk
                self._losses.append((actor_loss, critic_loss))
            self._publish_snapshot()
