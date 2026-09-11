"""Asynchronous actor-learner for the CPU-physics / GPU-learner split.

The envpool training loop is otherwise fully serial: it steps the CPU physics
pool, then runs the GPU gradient burst, each device idle while the other works.
Where the learner dominates (e.g. D4PG's distributional critic), that caps
throughput well below the CPU physics ceiling.

``AsyncLearner`` moves the gradient updates onto a background thread that is the
*sole owner* of ``agent.state``. The acting thread selects from a behaviour
actor snapshot the learner publishes, steps the envs on CPU, and hands
transitions over through a queue, so physics and gradient bursts overlap in
wall-clock time. That ownership split is also what makes the agents' fused
``_grad_steps`` donation of the whole train state safe: nothing else references
the buffer concurrently.

Semantics vs. the synchronous loop:
  * Replay ratio is preserved: the learner runs one ``learning_steps`` burst
    each time the number of *buffered* transitions crosses a
    ``steps_between_updates`` boundary past ``memory_warmup``. Behind, it runs
    bursts back-to-back; ahead, it waits for data.
  * The behaviour policy lags the learner by up to one burst, which an
    off-policy replay buffer tolerates.

Concurrency contract: the only cross-thread shared objects are the transition
queue and a lock-guarded snapshot/metrics slot, and the lock is held only for
reference swaps, never across device compute. It is NOT safe to call
``agent.step`` / ``agent.save`` while the learner runs; the trainer brackets
eval and checkpointing with ``pause()`` / ``resume()``.
"""

from __future__ import annotations

import functools
import queue
import threading

import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.utils import Transition


class AsyncLearner:
    # This learner's thread is the SOLE owner of `agent.state`, so a rollout
    # must go through `act`/`buffer` (the hand-off) and must NOT compile its own
    # acting against the train state. See `rollout.fusable`.
    owns_state = True

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
        # The acting thread's compiled `select_action`. See `_make_act_fn`.
        self._act_fn = None
        # The agent's exploration noise, if it has any. Owned by the ACTING
        # thread: the learner never reads it, and its decay counter has to
        # advance once per env frame acted.
        self._noise_module = getattr(agent, "noise_module", None)

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
        # Copied, not referenced: the `agent.learn` below donates these arrays.
        learner._behavior_stats = jax.tree.map(jnp.copy, agent.state.obs_stats)
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

    def _make_act_fn(self):
        """Behaviour-policy action selection, compiled.

        The acting thread's half of `EnvPoolRollout`'s fused step — and the only
        half this learner can have. **Buffering cannot follow it.** That half
        writes and DONATES `state.buffer_state` and `state.obs_stats`, which
        this learner's thread owns and donates again on every `_grad_steps`; an
        acting thread writing into them would not merely be racing, it would be
        writing to arrays a burst has already deleted. The queue is what stands
        in for it, and `EnvPoolRollout` falls back to the per-step loop whenever
        `fusable` sees a learner that `owns_state`.

        Acting has no such conflict. The behaviour actor, its frozen statistics
        and the noise module all belong to the ACTING thread — the learner only
        ever publishes a fresh snapshot for it to adopt — so the ten dispatches
        `select_action` used to issue can be one, exactly as on the sync path.

        No critic: only an on-policy agent needs one at acting time, and the
        async path is gated on a public `learn`, which only the off-policy
        agents expose. So this takes fewer leaves than the sync rollout's
        acting step.
        """
        agent = self._agent

        @functools.partial(jax.jit, donate_argnums=(1,))
        def act_fn(actor, noise_module, obs_stats, obs, key):
            action, applied_noise, _extras = agent.select_action(
                actor, obs_stats, obs, key, evaluate=False,
                noise_module=noise_module,
            )
            return noise_module, action, applied_noise

        return act_fn

    def act(self, obs, key):
        """Select actions from the behaviour snapshot — decoupled from the
        learner's live networks — re-syncing only when it advances.

        Extras (PPO's stored log-prob / value) are DROPPED, and the queue
        carries none: the async path is gated on a public `learn`, which only
        the extras-free off-policy agents expose.
        """
        version, snapshot = self._latest_snapshot()
        if version != self._behavior_version and snapshot is not None:
            nnx.update(self._behavior_actor, snapshot[0])
            self._behavior_stats = snapshot[1]
            self._behavior_version = version

        if self._act_fn is None:
            self._act_fn = self._make_act_fn()

        # The noise module is donated and re-adopted every step, because its
        # decay counter advances on every env frame acted. The rebind is also
        # published back to the agent, which is what a checkpoint reads.
        self._noise_module, action, applied_noise = self._act_fn(
            self._behavior_actor, self._noise_module, self._behavior_stats,
            obs, key,
        )
        if self._noise_module is not None:
            self._agent.noise_module = self._noise_module
        return action, applied_noise

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
        # The action came through the queue with this transition, rather than
        # off `agent.last_action`, which the acting thread overwrites every step.
        self._agent.buffer_transitions(
            Transition(
                observation=prev_obs,
                action=action,
                reward=reward,
                terminal=termination,
                truncation=truncation,
            ),
            next_obs,
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
        `_add_item` drains, so the learner never reaches its `learn` call.

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
