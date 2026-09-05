"""Asynchronous actor-learner for the CPU-physics / GPU-learner split.

The envpool training loop is otherwise fully serial: it steps the CPU physics
pool, then runs the GPU gradient burst, each device idle while the other works.
Where the learner dominates (e.g. D4PG's distributional critic), that caps
throughput well below the CPU physics ceiling.

``AsyncLearner`` moves the gradient updates onto a background thread that is the
*sole owner* of ``agent.state`` (networks, optimizers, replay buffer, obs
stats). The main (acting) thread never touches ``agent.state``: it selects
actions from a **behaviour actor** snapshot the learner publishes, steps the
envs on CPU, and hands the transitions to the learner through a queue, so the
physics and the gradient bursts overlap in wall-clock time.

That ownership split is what makes the agents' fused ``_grad_steps`` donation of
the whole train state safe: nothing else references the buffer concurrently. The
acting thread reads an independent behaviour actor and pushes raw transition
arrays; it never aliases the buffer.

Semantics vs. the synchronous loop:
  * Replay ratio is preserved: the learner runs one ``learning_steps`` burst
    each time the number of *buffered* transitions crosses a
    ``steps_between_updates`` boundary past ``memory_warmup``, the same
    cadence sync ``update`` uses. Behind, it runs bursts back-to-back; ahead, it
    waits for data — so the ratio matches sync on average and never runs ahead
    of collected data.
  * The behaviour policy lags the learner by up to one burst (standard for
    async off-policy RL), which an off-policy replay buffer tolerates.

Concurrency contract: the only cross-thread shared objects are the transition
queue (thread-safe) and a lock-guarded snapshot/metrics slot. The lock is held
only for cheap reference swaps, never across device compute, so JAX dispatch on
both threads can overlap. It is NOT safe to call ``agent.step`` / ``agent.save``
while the learner runs; the trainer brackets eval and checkpointing with
``pause()`` / ``resume()``.
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
        self.memory_warmup = int(agent.memory_warmup)
        self.steps_between_updates = int(agent.steps_between_updates)
        # `chunk` is how many fused grad steps the learner submits at a time —
        # small so the GPU stream keeps freeing up for the acting thread.
        # `_ratio` is the sync loop's grad-steps-per-env-step.
        self._chunk = max(1, min(int(chunk), self.learning_steps))
        self._ratio = self.learning_steps / max(1, self.steps_between_updates)

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._lock = threading.Lock()

        # Published behaviour snapshot (learner -> main), version-stamped so the
        # acting thread only re-syncs when it changes.
        self._snapshot = None            # (actor_param_state, obs_stats)
        self._snapshot_version = 0

        # Seeded from transitions already buffered synchronously (the warmup
        # fill) so the first burst fires on the same boundary the sync loop uses.
        self._added_steps = int(initial_steps)
        self._grad_steps = 0
        self._losses: list = []          # (actor_loss, critic_loss) since drain

        self._stop = threading.Event()
        self._pause_req = threading.Event()
        self._paused = threading.Event()   # learner acks it is idle/paused
        self._resume = threading.Event()
        self._resume.set()

        self._thread = threading.Thread(
            target=self._loop, name="async-learner", daemon=True
        )

        # Owned by the acting thread, held here so the snapshot-versioning
        # protocol stays this class's business.
        self._behavior_actor = None
        self._behavior_stats = None
        self._behavior_version = -1

    @classmethod
    def started(cls, agent, agent_key, state, *, initial_steps=0, chunk=8):
        """Build, warm the two programs the two threads will race on, and start.

        Both compiles happen HERE, before the learner exists: the acting
        thread's behaviour path and the learner's fixed-size chunk burst are new
        programs, and a cold compile of either mid-run would stall the other.
        """
        import copy

        agent_key, learner_key, chunk_key = jax.random.split(agent_key, 3)
        learner = cls(agent, learner_key, initial_steps=initial_steps, chunk=chunk)

        learner._behavior_actor = copy.deepcopy(agent.state.actor)
        learner._behavior_stats = agent.state.obs_stats
        warm_action, _, _ = agent.select_action(
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

    def start(self):
        # So the acting thread can act from step 0, off the warmup-era actor.
        self._publish_snapshot()
        self._thread.start()

    def act(self, obs, key):
        """Select actions from the behaviour snapshot — decoupled from the
        learner's live networks — re-syncing only when it advances."""
        version, snapshot = self._latest_snapshot()
        if version != self._behavior_version and snapshot is not None:
            nnx.update(self._behavior_actor, snapshot[0])
            self._behavior_stats = snapshot[1]
            self._behavior_version = version
        # Extras (PPO's stored log-prob / value) are DROPPED, and the queue
        # carries none: the async path is gated on a public `learn`, which only
        # the extras-free off-policy agents expose.
        action, noise, _extras = self._agent.select_action(
            self._behavior_actor, self._behavior_stats, obs, key, evaluate=False,
        )
        return action, noise

    def buffer(self, prev_obs, timestep, actions):
        """Hand the transition to the learner thread. The action travels WITH it
        rather than being read off `agent.last_action`, which `select_action`
        never sets.
        """
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

    def update(self, steps):
        """No-op: the learner thread paces its own bursts off the transitions it
        has buffered, not off the acting thread's step count. Present so the
        rollout can call the same surface on either learner."""
        del steps

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

    def _add_item(self, item) -> int:
        if item is None:  # stop / wake sentinel
            return 0
        prev_obs, action, reward, termination, truncation, next_obs = item
        self._agent.add_transitions(
            prev_obs, action, reward, termination, truncation, next_obs
        )
        return int(reward.shape[0])

    def _target_grads(self, added: int) -> float:
        """Cumulative grad steps the replay ratio owes for `added` buffered env
        steps: one full `learning_steps` burst at the `memory_warmup` boundary
        (as the sync loop does), then `ratio` more per env step."""
        if added < self.memory_warmup:
            return 0.0
        return self.learning_steps + self._ratio * (added - self.memory_warmup)

    def _drain_queue(self, bounded: bool = False) -> int:
        """Add queued transition batches to the buffer (the learner is the sole
        owner of the buffer).

        `bounded` STOPS as soon as the buffered data owes a full chunk of
        gradient work. Without it the loop exits only when the queue is
        momentarily empty, and when acting is cheaper than buffering (dm_control
        physics at 256 envs is nearly free) the acting thread refills faster than
        `_add_item` drains, so the learner never reaches its `learn` call. That
        livelock neither deadlocks nor raises: the run just finishes at a
        fraction of its replay ratio, silently untrained — DDPG/AcrobotSwingup
        took 40 of the 2,640 gradient steps the schedule asks for.

        `pause()` still wants the unbounded form: there the acting thread is
        quiesced, so the queue does drain, and stranded data would be lost.
        """
        added = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            added += self._add_item(item)
            if bounded and (
                self._target_grads(self._added_steps + added) - self._grad_steps
                >= self._chunk
            ):
                break
        if added:
            with self._lock:
                self._added_steps += added
        return added

    def _publish_snapshot(self):
        # Independent on-device copies: a plain reference would alias buffers
        # the next `learn()` donates, leaving the acting thread reading freed
        # memory. The copy is issued before that donation on the same stream.
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

            self._drain_queue(bounded=True)

            with self._lock:
                added = self._added_steps
                grads = self._grad_steps

            target = self._target_grads(added)

            if grads >= target:
                # Ahead of schedule: wait for data rather than busy-spin and
                # starve the acting thread's cores.
                try:
                    item = self._queue.get(timeout=0.005)
                except queue.Empty:
                    continue
                n = self._add_item(item)
                if n:
                    with self._lock:
                        self._added_steps += n
                continue

            # Fixed-size, so `_grad_steps` compiles once (n_steps is static).
            self._key, burst_key = jax.random.split(self._key)
            actor_loss, critic_loss = self._agent.learn(
                burst_key, n_steps=self._chunk
            )
            with self._lock:
                self._grad_steps += self._chunk
                self._losses.append((actor_loss, critic_loss))
            self._publish_snapshot()
