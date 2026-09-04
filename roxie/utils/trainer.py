import functools
import os
import time

import numpy as np
import jax
import orbax.checkpoint as ocp
from flax import nnx

from roxie.utils import logger
from roxie.utils.checkpoint import CHECKPOINTS_DIRNAME
from roxie.utils.learner import build_learner
from roxie.utils.rollout import CHUNK_SUMS, build_rollout, timed


def _new_epoch_acc():
    """Per-epoch accumulators, reset at every epoch boundary.

    The `CHUNK_SUMS` half is what `rollout.collect` returns per chunk and this
    only sums up; `steps` counts the env steps those sums cover, which is what
    the per-step means divide by. The loss lists stay host-side: they are one
    entry per gradient burst, not per step.
    """
    return dict(
        {key: 0.0 for key in CHUNK_SUMS},
        metrics={}, steps=0, actor_losses=[], critic_losses=[],
    )


def _merge_chunk(acc, sums, steps):
    """Fold one collected chunk into the epoch accumulator."""
    for key in CHUNK_SUMS:
        acc[key] = acc[key] + sums[key]
    for key, value in sums["metrics"].items():
        acc["metrics"][key] = acc["metrics"].get(key, 0.0) + value
    acc["steps"] += steps


class Trainer:
    """One training loop over every backend.

    What varies between a vmapped JAX env and a C++ EnvPool pool is behind
    `roxie.utils.rollout`; whether learning runs inline or on a background thread
    is behind `roxie.utils.learner`. What is left here is step budget, epoch and
    save cadences, episode statistics and metric namespacing.
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
        self.save_buffer = bool(save_buffer)
        resume = resume or {}
        (self.initial_steps, self.initial_epochs, self.initial_episodes,
         self.initial_gradient_steps) = (
            int(resume.get(k) or 0)
            for k in ("steps", "epochs", "episodes", "gradient_steps")
        )
        self.skip_warmup = bool(resume.get("buffer_restored", False))
        # `learner_chunk` = fused grad steps per dispatch: smaller overlaps
        # better with acting, larger amortizes dispatch overhead.
        self.async_learner = async_learner
        self.learner_chunk = int(learner_chunk)
        self._checkpoint_manager = None

    def initialize(self, agent, environment, test_environment=None):
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    def _warmup_iters(self, memory_warmup, num_envs):
        """Zero when a resume restored a populated buffer — refilling would add
        random-action transitions to a trained policy's data."""
        if self.skip_warmup:
            print("Resumed with a restored replay buffer: skipping warmup.",
                  flush=True)
            return 0
        return memory_warmup // num_envs

    def _checkpoints(self):
        """The run's `checkpoints/` manager, opened on first save.

        Saves are asynchronous — `checkpoint_payload` hands over host arrays, so
        the write proceeds off the training loop and `close()` waits for it.
        """
        if self._checkpoint_manager is None:
            self._checkpoint_manager = ocp.CheckpointManager(
                os.path.join(self.output_dir, CHECKPOINTS_DIRNAME),
                options=ocp.CheckpointManagerOptions(
                    max_to_keep=1 if self.replace_checkpoint else None,
                    create=True,
                ),
            )
        return self._checkpoint_manager

    def _save(self, agent, epochs, episodes, gradient_steps):
        """Write a checkpoint, tagged with the progress needed to resume it."""
        payload = agent.checkpoint_payload(
            include_buffer=self.save_buffer,
            extra_metadata={
                "steps": int(self.steps),
                "epochs": int(epochs),
                "episodes": int(episodes),
                "gradient_steps": int(gradient_steps),
            },
        )
        if payload is None:  # a baseline with nothing to checkpoint
            return
        self._checkpoints().save(
            int(self.steps), args=ocp.args.StandardSave(payload)
        )

    def _resync_cadence(self):
        """Re-phase the epoch and save cadences against the current `self.steps`.
        Called whenever `steps` jumps rather than increments (resume, warmup fill)."""
        return self.steps % self.epoch_steps, self.steps % self.save_steps

    def _init_counters(self):
        """Seed the run counters from the resume metadata (all zeros for a fresh
        run), so `steps` keeps meaning total env steps across a restart."""
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
        # Hand back the boundary this call consumed, so the loop runs the full
        # schedule and the realized replay ratio matches the config.
        agent._last_update_boundary = -1
        print(f"  {time.time() - t0:.1f}s", flush=True)
        return agent_key

    @staticmethod
    def _collect_steps(agent, num_envs):
        """Env steps per `rollout.collect` call — the loop's unit of work.

        ONE `steps_between_updates` window, so the chunk ends exactly where the
        gradient burst fires and the actor is constant for its whole duration.
        That is what makes the fused acting burst a throughput change and not an
        algorithm change: the per-step loop also left the actor untouched
        between bursts, so the two produce the same schedule.

        An agent that does not declare the cadence (PPO, which updates when its
        rollout buffer fills, and the baselines) gets 1 — today's per-step loop,
        unchanged.
        """
        between = int(getattr(agent, "steps_between_updates", 0) or 0)
        return max(1, between // int(num_envs))

    @staticmethod
    def _bench(bench_steps, bench_t0, obs):
        """One throughput print, after the first few chunks. Blocks on the
        loop's output first, so this times real work rather than dispatch."""
        jax.block_until_ready(obs)
        elapsed = time.time() - bench_t0
        print(
            f"\n  Speed: {bench_steps / elapsed:.0f} steps/s",
            flush=True,
        )

    def run(self, NUM_ENVS, rngs):
        """Build the backend rollout and the learning strategy, then train."""
        rollout = build_rollout(
            self.environment, self.test_environment, self.agent,
            NUM_ENVS, rngs, self.test_episodes,
        )
        try:
            self._run(rollout, NUM_ENVS, rngs)
        finally:
            # Saves are asynchronous, so the last one is still in flight here.
            if self._checkpoint_manager is not None:
                self._checkpoint_manager.close()

    def _run(self, rollout, NUM_ENVS, rngs):
        start_time = last_epoch_time = time.time()
        agent = self.agent
        # Read by `_store_epoch_metrics`, which turns the rollout's per-step
        # sums back into per-step means.
        self.num_envs = int(NUM_ENVS)
        agent_key = rngs.agent()

        state = rollout.prepare()
        agent_key, compile_key = jax.random.split(agent_key)
        timed(functools.partial(agent.step, evaluate=False, key=compile_key),
              state.obs, label="agent step")

        memory_warmup = getattr(agent, "memory_warmup", 0)
        warmup_iters = self._warmup_iters(memory_warmup, NUM_ENVS)
        (epochs, episodes, tot_gradient_steps,
         epoch_steps, steps_since_save) = self._init_counters()

        if warmup_iters > 0:
            print(f"Warmup: {warmup_iters} iters ({memory_warmup:,} steps)...",
                  flush=True)
            state, warmup_episodes = rollout.warmup(agent, warmup_iters, state)
            # Warmup transitions are real env steps, so they count.
            self.steps = self.initial_steps + warmup_iters * NUM_ENVS
            epoch_steps, steps_since_save = self._resync_cadence()
            episodes += warmup_episodes
            print(f"  Done: {self.steps:,} steps, {int(episodes)} episodes",
                  flush=True)

        agent_key = self._precompile_update(agent, agent_key)

        learner = build_learner(self, rollout, agent, agent_key, state)

        print("Training...", flush=True)
        # One update window per iteration, so every cadence measured in env
        # steps is quantized to this chunk. The CSV records the actual `steps`.
        collect_steps = self._collect_steps(agent, NUM_ENVS)
        chunk_env_steps = collect_steps * NUM_ENVS
        acc = _new_epoch_acc()
        bench_t0, bench_steps = time.time(), 0

        while True:
            state, sums = rollout.collect(state, collect_steps, learner)
            self.steps += chunk_env_steps
            epoch_steps += chunk_env_steps
            steps_since_save += chunk_env_steps
            bench_steps += chunk_env_steps
            _merge_chunk(acc, sums, chunk_env_steps)
            # Left as a device scalar on the JAX path and reduced to an int
            # only at the epoch boundary; `int()` here would sync every chunk.
            episodes = episodes + sums["count"]

            learner.update(self.steps)
            grad_steps, new_losses = learner.drain()
            tot_gradient_steps = self.initial_gradient_steps + grad_steps
            for actor_loss, critic_loss in new_losses:
                acc["actor_losses"].append(actor_loss)
                acc["critic_losses"].append(critic_loss)

            if bench_steps >= NUM_ENVS * 20 and bench_t0 is not None:
                self._bench(bench_steps, bench_t0, state.obs)
                bench_t0 = None

            if self.show_progress:
                logger.show_progress(self.steps, self.epoch_steps, self.max_steps)

            if epoch_steps >= self.epoch_steps:
                epoch_steps = 0
                epochs, acc = self._end_of_epoch(
                    agent, acc, rollout, learner,
                    epochs=epochs, episodes=episodes,
                    gradient_steps=tot_gradient_steps,
                    start_time=start_time, last_epoch_time=last_epoch_time,
                )
                # After the metrics are stored: the refresh is where the env
                # changes itself, and this epoch's numbers describe it as it was.
                state, invalidated = rollout.epoch_refresh(state)
                if invalidated:
                    # The live envs were reset, so the part-scored episodes in
                    # flight are meaningless.
                    rollout.reset_tally()
                last_epoch_time = time.time()

            steps_since_save, stop_training = self._checkpoint_if_due(
                agent, learner, epochs=epochs, episodes=episodes,
                gradient_steps=tot_gradient_steps,
                steps_since_save=steps_since_save,
            )
            if stop_training:
                break

        learner.stop()

    def _end_of_epoch(self, agent, acc, rollout, learner, *,
                      epochs, episodes, gradient_steps, start_time,
                      last_epoch_time):
        """The epoch boundary: reset exploration noise, run the held-out eval,
        store and dump the epoch's metrics. Returns `(epochs, acc)`.

        The backend's own refresh is the caller's next move — only it knows which
        loop state that invalidates.
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
            agent, self._epoch_stats(acc, rollout.scores, rollout.lengths), acc,
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

    @staticmethod
    def _epoch_stats(acc, scores, lengths):
        """(n, mean/std of return, mean/std of length) over episodes that
        completed this epoch, falling back to the in-flight per-env counters when
        none did (episodes longer than an epoch) so the line is never blank."""
        n = int(acc["count"])
        scores, lengths = np.asarray(scores), np.asarray(lengths)
        if n == 0:
            return (n, float(np.mean(scores)), float(np.std(scores)),
                    float(np.mean(lengths)), float(np.std(lengths)))
        # Dividing by the accumulator rather than `n` keeps the arithmetic in
        # its own dtype (f32 on the jax path) instead of widening.
        mean_ret = float(acc["ret"] / acc["count"])
        mean_len = float(acc["len"] / acc["count"])

        # Clamped against tiny negative variances from float round-off.
        def std(sq, mean):
            return float(np.sqrt(max(float(acc[sq] / acc["count"]) - mean ** 2, 0.0)))

        return (n, mean_ret, std("ret_sq", mean_ret),
                mean_len, std("len_sq", mean_len))

    def _store_epoch_metrics(
        self, agent, stats, acc, *, epochs, episodes, sps, gradient_steps,
        times,
    ):
        """Store one epoch's metrics under the shared namespace scheme.

        ``epoch``/``steps`` are the run axes, ``train/*`` the behaviour policy
        and learner, ``test/*`` the held-out eval, ``sys/*`` throughput and
        host/device health. Anything a producer already prefixes for itself
        (``reward/``, ``noise/``, and each agent's own name from
        ``pop_diagnostics``) nests under ``train/``.
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
        # None, not 0.0, when the epoch ran no gradient bursts: a logged zero
        # is indistinguishable from a converged loss, and renders as a gap.
        for name in ("actor", "critic"):
            values = acc[f"{name}_losses"]
            logger.store(f"train/loss/{name}",
                         float(np.mean(values)) if values else None)

        # The rollout already reduced each step to a scalar, so this divides by
        # the number of steps those scalars came from.
        iters = acc["steps"] / self.num_envs
        extra = {k: float(v / iters) for k, v in acc["metrics"].items()} if iters else {}
        # Post-clip, in normalized [-1, 1] action units; compare against the
        # noise module's scheduled scale to see how much clipping eats.
        if iters:
            extra["noise/per_joint_abs"] = float(acc["noise"] / iters)
        # Every agent answers this; the ones with nothing to report answer {}.
        extra.update(agent.pop_diagnostics(self.steps))
        for k, v in extra.items():
            logger.store(f"train/{k}", float(v))

        now = time.time()
        logger.store("sys/sps", sps)
        logger.store("sys/time/total_s", now - start_time)
        logger.store("sys/time/epoch_s", now - last_epoch_time)
        # Whole-card telemetry: a run not on the GPU would attribute another
        # process's memory and utilisation to itself.
        if any(d.platform == "gpu" for d in jax.devices()):
            for k, v in logger.gpu_stats().items():
                logger.store(k, v)
        # /proc is Linux-only.
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
        # Episode return is a sum, so with spread start phases it is bounded by
        # how much episode was left at reset; the per-step rate divides that out.
        logger.store("test/score_per_step",
                     float(np.mean(scores / np.maximum(lengths, 1))))
        # `test/length/std` cannot stand in: 0.00 there means either a
        # degenerate eval or a policy saturating the episode cap.
        logger.store("test/distinct_starts",
                     float(len(np.unique(start_obs, axis=0))))
