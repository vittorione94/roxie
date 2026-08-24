import functools
import os
import shutil
import time

import numpy as np
import jax
from flax import nnx

from roxie.utils import logger
from roxie.utils.learner import build_learner
from roxie.utils.rollout import build_rollout, timed


def _new_epoch_acc():
    """Per-epoch accumulators, reset at every epoch boundary.

    `ret`/`len` collect only episodes that actually terminated; their `_sq`
    companions recover the per-episode std at the boundary via
    sqrt(E[x^2] - E[x]^2), avoiding a host-side list (and a per-step sync).
    `metrics` sums the env's own metric dict, keys discovered lazily. `noise`
    needs its own counter: it is added on policy steps only, not during warmup.
    `actor_losses`/`critic_losses` collect one entry per gradient burst and are
    reduced host-side at the boundary, so an epoch that ran none logs a gap
    rather than a zero. Zeros are plain floats — jnp promotes them on first use.
    """
    return dict(
        ret=0.0, ret_sq=0.0, len=0.0, len_sq=0.0, count=0.0,
        metrics={}, metric_iters=0, noise=0.0, noise_iters=0,
        actor_losses=[], critic_losses=[],
    )


class Trainer:
    """One training loop over every backend.

    What varies between a vmapped JAX env and a C++ EnvPool pool is behind
    `roxie.utils.rollout`; whether learning runs inline or on a background thread
    is behind `roxie.utils.learner`. What is left here — step budget, epoch and
    save cadences, episode statistics, metric namespacing — is the same
    arithmetic regardless, and is written once.
    """

    def __init__(
        self, output_dir, steps=int(1e7), epoch_steps=int(3e5), save_steps=int(1e5),
        test_episodes=5, show_progress=True, replace_checkpoint=False,
        async_learner=False, learner_chunk=8, save_buffer=False, resume=None,
    ):
        self.max_steps = steps
        self.epoch_steps = epoch_steps
        self.save_steps = save_steps
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint
        self.output_dir = output_dir
        # Off by default: the buffer dominates checkpoint size and the host-RAM
        # cost of writing it. On, a resumed off-policy run skips warmup.
        self.save_buffer = bool(save_buffer)
        # Resume counters keep a restarted run on the SAME curve: env steps keyed
        # identically, save cadence and step budget measured against the total.
        resume = resume or {}
        (self.initial_steps, self.initial_epochs, self.initial_episodes,
         self.initial_gradient_steps) = (
            int(resume.get(k) or 0)
            for k in ("steps", "epochs", "episodes", "gradient_steps")
        )
        # Only set when the run that wrote the checkpoint had `save_buffer`;
        # otherwise a resumed agent would sample a buffer of zeros.
        self.skip_warmup = bool(resume.get("buffer_restored", False))
        # Background learner thread overlapping CPU acting with GPU gradient
        # bursts. `learner_chunk` = fused grad steps per dispatch: smaller
        # overlaps better, larger amortizes dispatch overhead.
        self.async_learner = async_learner
        self.learner_chunk = int(learner_chunk)

    def initialize(self, agent, environment, test_environment=None):
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    def _warmup_iters(self, memory_warmup, num_envs):
        """Zero when a resume restored a populated buffer — refilling would
        prepend a block of RANDOM-action transitions to a trained policy's data."""
        if self.skip_warmup:
            print("Resumed with a restored replay buffer: skipping warmup.",
                  flush=True)
            return 0
        return memory_warmup // num_envs

    def _save(self, agent, epochs, episodes, gradient_steps):
        """Write a checkpoint, tagged with the progress needed to resume it."""
        path = os.path.join(self.output_dir, "checkpoints")
        if os.path.isdir(path) and self.replace_checkpoint:
            for file in os.listdir(path):
                if file.startswith("step_"):
                    # A checkpoint is a DIRECTORY (orbax writes a tree), so this
                    # needs rmtree — `os.remove` raises IsADirectoryError.
                    shutil.rmtree(os.path.join(path, file), ignore_errors=True)
        agent.save(
            os.path.join(path, f"step_{self.steps}"),
            include_buffer=self.save_buffer,
            extra_metadata={
                "steps": int(self.steps),
                "epochs": int(epochs),
                "episodes": int(episodes),
                "gradient_steps": int(gradient_steps),
            },
        )

    # ------------------------------------------------------------------
    # Counters, cadences and accumulators
    # ------------------------------------------------------------------

    def _resync_cadence(self):
        """Re-phase the epoch and save cadences against the current `self.steps`.
        Called whenever `steps` jumps rather than increments (resume, warmup fill)."""
        return self.steps % self.epoch_steps, self.steps % self.save_steps

    def _init_counters(self):
        """Seed the run counters from the resume metadata (all zeros for a fresh
        run), so `steps` keeps meaning TOTAL env steps: epoch and save cadences
        stay phase-aligned across a restart and the logs continue one curve."""
        self.steps = self.initial_steps
        epoch_steps, steps_since_save = self._resync_cadence()
        return (self.initial_epochs, self.initial_episodes,
                self.initial_gradient_steps, epoch_steps, steps_since_save)

    def _precompile_update(self, agent, agent_key):
        """Compile the gradient step once the buffer holds warmup data. Returns
        the advanced key. No-op for agents without the update contract."""
        if not (hasattr(agent, "update") and hasattr(agent, "steps_before_learning")):
            return agent_key
        print("Compiling gradient step...", flush=True)
        t0 = time.time()
        agent_key, warm_key = jax.random.split(agent_key)
        agent.update(steps=agent.steps_before_learning, agent_rng=warm_key)
        jax.block_until_ready(jax.tree.leaves(nnx.state(agent.state)))
        # Hand back the boundary that call consumed, so the loop runs the full
        # schedule and the realized replay ratio is the one the config asks for.
        agent._last_update_boundary = -1
        print(f"  {time.time() - t0:.1f}s", flush=True)
        return agent_key

    @staticmethod
    def _accumulate_episodes(acc, scores, lengths, done, xp):
        """Fold this step's COMPLETED episodes into the epoch accumulators and
        return how many finished.

        `scores`/`lengths` already include the terminal transition and are zeroed
        for done envs by the caller, so non-done envs contribute 0 here. `xp` is
        the array namespace: `jnp` on the JAX rollout, where every op must stay on
        device (a host reduction here would sync the pipeline every step), `np`
        on the EnvPool rollout, where the arrays are already host-side. The ops
        are spelled identically in both.
        """
        done_f = done.astype(xp.float32)
        done_sum = xp.sum(done_f)  # once — reused for `episodes` and `count`
        ep_ret, ep_len = scores * done_f, lengths.astype(xp.float32) * done_f
        acc["ret"] += xp.sum(ep_ret)
        acc["ret_sq"] += xp.sum(ep_ret ** 2)
        acc["len"] += xp.sum(ep_len)
        acc["len_sq"] += xp.sum(ep_len ** 2)
        acc["count"] += done_sum
        return done_sum

    @staticmethod
    def _bench(bench_steps, bench_t0, obs):
        """One throughput print, after the first 20 iterations. Blocks on the
        loop's output first, so this times real work rather than dispatch."""
        jax.block_until_ready(obs)
        elapsed = time.time() - bench_t0
        print(
            f"\n  Speed: {bench_steps / elapsed:.0f} steps/s "
            f"({elapsed / 20 * 1000:.1f}ms/iter)",
            flush=True,
        )

    # ------------------------------------------------------------------
    # The training loop
    # ------------------------------------------------------------------

    def run(self, NUM_ENVS, rngs):
        """Build the backend rollout and the learning strategy, then train.

        Which backend this is, and whether learning is synchronous, is decided
        here and nowhere else — `_run` below never asks.
        """
        rollout = build_rollout(
            self.environment, self.test_environment, self.agent,
            NUM_ENVS, rngs, self.test_episodes,
        )
        self._run(rollout, NUM_ENVS, rngs)

    def _run(self, rollout, NUM_ENVS, rngs):
        start_time = last_epoch_time = time.time()
        agent = self.agent
        xp = rollout.xp
        agent_key = rngs.agent()

        state = rollout.prepare()
        agent_key, compile_key = jax.random.split(agent_key)
        timed(functools.partial(agent.step, evaluate=False, key=compile_key),
              state.obs, label="agent step")

        # --- Warmup: fill the replay buffer with random actions ---

        memory_warmup = getattr(agent, "memory_warmup", 0)
        warmup_iters = self._warmup_iters(memory_warmup, NUM_ENVS)
        (epochs, episodes, tot_gradient_steps,
         epoch_steps, steps_since_save) = self._init_counters()

        if warmup_iters > 0:
            print(f"Warmup: {warmup_iters} iters ({memory_warmup:,} steps)...",
                  flush=True)
            state, warmup_episodes = rollout.warmup(agent, warmup_iters, state)
            # Warmup transitions are real env steps, so they count — on a resume
            # they add on top of the restored total rather than replacing it.
            self.steps = self.initial_steps + warmup_iters * NUM_ENVS
            epoch_steps, steps_since_save = self._resync_cadence()
            episodes += warmup_episodes
            print(f"  Done: {self.steps:,} steps, {int(episodes)} episodes",
                  flush=True)

        # Precompile the gradient step (the buffer now holds warmup data).
        agent_key = self._precompile_update(agent, agent_key)

        # The warmup fill is the ONLY random-action phase: it runs exactly
        # `warmup_iters * NUM_ENVS` steps and the loop below is always on-policy.
        # (The JAX loop used to additionally gate on `steps > memory_warmup`,
        # which the EnvPool loop never did, so the two collected different
        # numbers of random steps whenever `memory_warmup % NUM_ENVS != 0`.)
        learner = build_learner(self, rollout, agent, agent_key, state)

        # --- Training loop ---

        print("Training...", flush=True)
        # In-flight per-env counters; `acc` collects only completed episodes.
        scores = xp.zeros(NUM_ENVS)
        lengths = xp.zeros(NUM_ENVS, dtype=xp.int32)
        acc = _new_epoch_acc()
        bench_t0, bench_steps = time.time(), 0
        action_rng = rngs.envs()

        while True:
            action_rng, action_key = jax.random.split(action_rng)
            actions, last_noise = learner.act(state.obs, action_key)
            # Mean |noise| per joint in normalized action units. Only
            # deterministic agents (DDPG/TD3) expose it.
            if last_noise is not None:
                acc["noise"] += xp.mean(xp.abs(last_noise))
                acc["noise_iters"] += 1

            state, prev_obs, timestep = rollout.step(state, actions)
            learner.observe(prev_obs, timestep, actions, self.steps)

            grad_steps, new_losses = learner.drain()
            tot_gradient_steps = self.initial_gradient_steps + grad_steps
            for actor_loss, critic_loss in new_losses:
                acc["actor_losses"].append(actor_loss)
                acc["critic_losses"].append(critic_loss)

            reward = xp.asarray(timestep.reward)
            # An episode ends on EITHER flag; the two are kept apart only for the
            # Bellman bootstrap, which is the agent's business, not the
            # bookkeeping's.
            done = xp.asarray(timestep.terminated) | xp.asarray(timestep.truncated)
            scores = scores + reward
            lengths = lengths + 1
            # Per-env vectors; the mean is deferred to the epoch boundary to
            # avoid one dispatch per metric per step.
            for k, v in timestep.info.get("metrics", {}).items():
                acc["metrics"][k] = acc["metrics"].get(k, 0.0) + v
            acc["metric_iters"] += 1

            self.steps += NUM_ENVS
            epoch_steps += NUM_ENVS
            steps_since_save += NUM_ENVS
            bench_steps += NUM_ENVS

            episodes = episodes + self._accumulate_episodes(
                acc, scores, lengths, done, xp,
            )

            if bench_steps == NUM_ENVS * 20:
                self._bench(bench_steps, bench_t0, state.obs)

            if self.show_progress:
                logger.show_progress(self.steps, self.epoch_steps, self.max_steps)

            if epoch_steps >= self.epoch_steps:
                epoch_steps = 0
                epochs, acc = self._end_of_epoch(
                    agent, acc, scores, lengths, rollout, learner,
                    epochs=epochs, episodes=episodes,
                    gradient_steps=tot_gradient_steps,
                    start_time=start_time, last_epoch_time=last_epoch_time,
                )
                # After the metrics are stored: the refresh is where the env
                # gets to change itself, and this epoch's numbers describe the
                # env as it was.
                state, invalidated = rollout.epoch_refresh(state)
                if invalidated:
                    # The backend reset the live envs, so the part-scored
                    # episodes in flight are no longer meaningful.
                    scores = xp.zeros(NUM_ENVS)
                    lengths = xp.zeros(NUM_ENVS, dtype=xp.int32)
                last_epoch_time = time.time()

            scores = xp.where(done, 0, scores)
            lengths = xp.where(done, 0, lengths)

            steps_since_save, stop_training = self._checkpoint_if_due(
                agent, learner, epochs=epochs, episodes=episodes,
                gradient_steps=tot_gradient_steps,
                steps_since_save=steps_since_save,
            )
            if stop_training:
                break

        learner.stop()

    def _end_of_epoch(self, agent, acc, scores, lengths, rollout, learner, *,
                      epochs, episodes, gradient_steps, start_time,
                      last_epoch_time):
        """The epoch boundary: reset exploration noise, run the held-out eval,
        store and dump the epoch's metrics, hand back a fresh accumulator.

        Whatever the backend must refresh afterwards (reset pool, plus whatever
        the env changes about itself) is the caller's next move — only it knows
        which loop state that invalidates. Returns `(epochs, acc)`.
        """
        if hasattr(agent, "noise_module"):
            agent.noise_module.reset_noise()

        if self.test_environment and hasattr(agent, "state"):
            # Eval reads agent.state; quiesce the learner so it observes a
            # consistent state and doesn't contend for the device.
            learner.pause()
            test_scores, test_lengths, start_obs = rollout.evaluate(agent)
            learner.resume()
            self._store_test_metrics(test_scores, test_lengths, start_obs)

        epochs += 1
        self._store_epoch_metrics(
            agent, self._epoch_stats(acc, scores, lengths), acc,
            epochs=epochs, episodes=episodes,
            sps=self.steps / (time.time() - start_time),
            gradient_steps=gradient_steps,
            times=(start_time, last_epoch_time),
        )
        logger.dump(step=self.steps)
        return epochs, _new_epoch_acc()

    def _checkpoint_if_due(self, agent, learner, *, epochs, episodes,
                           gradient_steps, steps_since_save):
        """Save on the save cadence and at the step budget. Checkpointing reads
        (and momentarily detaches) `agent.state`, so the learner is quiesced
        first — a background one must not grad-step a half-detached state.
        Returns `(steps_since_save, stop_training)`.
        """
        stop_training = self.steps >= self.max_steps
        if stop_training or steps_since_save >= self.save_steps:
            learner.pause()
            self._save(agent, epochs, episodes, gradient_steps)
            learner.resume()
            steps_since_save = self.steps % self.save_steps
        return steps_since_save, stop_training

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _epoch_stats(acc, scores, lengths):
        """(n, mean/std of return, mean/std of length) over episodes that
        COMPLETED this epoch, falling back to the in-flight per-env counters when
        none did (episodes longer than an epoch) so the line is never blank."""
        n = int(acc["count"])
        if n == 0:
            return (n, float(np.mean(scores)), float(np.std(scores)),
                    float(np.mean(lengths)), float(np.std(lengths)))
        # Divide by the accumulator, not by `n`: that keeps the arithmetic in the
        # accumulator's own dtype (f32 on the jax path) instead of widening first.
        mean_ret = float(acc["ret"] / acc["count"])
        mean_len = float(acc["len"] / acc["count"])

        # Population std, clamped against tiny negative variances from float
        # round-off in the E[x^2] - E[x]^2 subtraction.
        def std(sq, mean):
            return float(np.sqrt(max(float(acc[sq] / acc["count"]) - mean ** 2, 0.0)))

        return (n, mean_ret, std("ret_sq", mean_ret),
                mean_len, std("len_sq", mean_len))

    def _store_epoch_metrics(
        self, agent, stats, acc, *, epochs, episodes, sps, gradient_steps,
        times,
    ):
        """Store one epoch's metrics under the shared namespace scheme.

        Every metric is namespaced by its producer: ``epoch``/``steps`` are the
        run axes, ``train/*`` the behaviour policy and learner, ``test/*`` the
        held-out eval, ``sys/*`` throughput and host/device health. Anything a
        producer already prefixes for itself (``reward/``, ``noise/``, ``td3/``,
        ``ppo/``) nests UNDER ``train/`` — this is the only place that knows the
        namespace.
        """
        ep_n, score, score_std, length, length_std = stats
        start_time, last_epoch_time = times

        logger.store("epoch", epochs)
        logger.store("steps", self.steps)

        logger.store("train/score", score)
        logger.store("train/score/std", score_std)
        logger.store("train/length", length)
        logger.store("train/length/std", length_std)
        logger.store("train/episodes/epoch", ep_n)
        logger.store("train/episodes/total", int(episodes))
        logger.store("train/gradient_steps", gradient_steps)
        # None, not 0.0, when the epoch ran no gradient bursts: a logged zero is
        # indistinguishable from a real converged loss, and that ambiguity hid the
        # v1 release grid's zero-gradient-step bug for a full overnight sweep.
        # Backends render an absent metric as a gap (see `WandbBackend.log`).
        for name in ("actor", "critic"):
            values = acc[f"{name}_losses"]
            logger.store(f"train/loss/{name}",
                         float(np.mean(values)) if values else None)

        # Epoch-mean of each env metric, e.g. `train/reward/upright`. Per-env
        # vectors (jax rollout) and scalars (envpool rollout) both reduce via
        # np.mean. (`float()` last, so the division keeps the accumulator's dtype.)
        extra = {k: float(np.mean(v) / acc["metric_iters"])
                 for k, v in acc["metrics"].items()} if acc["metric_iters"] else {}
        # Post-clip noise in normalized [-1, 1] action units; compare against the
        # noise module's scheduled scale to see how much clipping eats.
        if acc["noise_iters"]:
            extra["noise/per_joint_abs"] = float(acc["noise"] / acc["noise_iters"])
        # Optional per-agent diagnostics (TD3 saturation/value, PPO trust region).
        diagnostics = getattr(agent, "pop_diagnostics", None)
        extra.update(diagnostics() if diagnostics is not None else {})
        for k, v in extra.items():
            logger.store(f"train/{k}", float(v))

        now = time.time()
        logger.store("sys/sps", sps)
        logger.store("sys/time/total_s", now - start_time)
        logger.store("sys/time/epoch_s", now - last_epoch_time)
        # Device telemetry is WHOLE-CARD, so a run not on the GPU would attribute
        # another process's memory and utilisation to itself. No-op without
        # nvidia-smi.
        if any(d.platform == "gpu" for d in jax.devices()):
            for k, v in logger.gpu_stats().items():
                logger.store(k, v)
        # Host-memory watch, so creep toward an OOM is visible. /proc is Linux-only.
        try:
            rss_pages = int(open("/proc/self/statm").read().split()[1])
            logger.store("sys/mem/rss_gb",
                         rss_pages * os.sysconf("SC_PAGE_SIZE") / 1e9)
            logger.store("sys/mem/live_arrays", len(jax.live_arrays()))
        except (OSError, ValueError):
            pass

    @staticmethod
    def _store_test_metrics(scores, lengths, start_obs):
        """Stored, not printed: the eval runs just before `logger.dump()`, so its
        stats land in the same dump as the training stats."""
        logger.store("test/score", float(np.mean(scores)))
        logger.store("test/score/std", float(np.std(scores)))
        logger.store("test/length", float(np.mean(lengths)))
        logger.store("test/length/std", float(np.std(lengths)))
        # Episode return is a SUM, so with spread start phases it is bounded by how
        # much clip was left at reset: a policy starting late in a non-cyclic clip
        # cannot score what a frame-0 start can, however well it tracks. The
        # per-step rate divides that out and is start-phase invariant.
        logger.store("test/score_per_step",
                     float(np.mean(scores / np.maximum(lengths, 1))))
        # How many genuinely distinct states the eval batch starts from — the env
        # decides its own reset stochasticity in Python and none of that reaches
        # the logged config. `test/length/std` cannot stand in: 0.00 there means a
        # degenerate eval OR a policy saturating the episode cap, opposite news.
        logger.store("test/distinct_starts",
                     float(len(np.unique(start_obs, axis=0))))
