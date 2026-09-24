"""Trainer utility for managing the training loop, checkpoints, and epoch metrics."""
import os
import time

import numpy as np
import jax
import orbax.checkpoint as ocp
from flax import nnx

from roxie.agents.hyperparams import OffPolicyHyperparams
from roxie.utils import logger
from roxie.utils.checkpoint import CHECKPOINTS_DIRNAME
from roxie.utils.rollout import ChunkSums, RolloutState, build_rollout


def _new_epoch_acc(metric_keys):
    """Per-epoch accumulators, reset at every epoch boundary.

    `sums` is the `ChunkSums` `rollout.collect` returns per chunk, summed; it
    stays on device until the boundary reads it. `steps` counts the env steps
    those sums cover, which is what the per-step means divide by. The loss lists
    stay host-side: they are one entry per learning pass, not per step.
    """
    return dict(
        sums=ChunkSums.zeros(metric_keys),
        steps=0, actor_losses=[], critic_losses=[],
    )


class Trainer:
    """One training loop over every backend.

    What varies between a vmapped JAX env and a C++ EnvPool pool is behind
    `roxie.utils.rollout`, which fuses a whole update window of acting,
    physics, buffering and scoring into one compiled chunk. What is left here is
    the step budget, the epoch and save cadences, the learning pass's own
    bookkeeping, episode statistics and metric namespacing.
    """

    def __init__(
        self, output_dir, steps=int(1e7), epoch_steps=int(3e5), save_steps=int(1e5),
        test_episodes=5, show_progress=True, replace_checkpoint=False,
        save_buffer=False, resume=None,
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

    @staticmethod
    def _crossed(prev_steps, steps, stride):
        """Did the step count cross a multiple of `stride` this chunk?

        Both cadences are locked to the ABSOLUTE step grid rather than counted
        up from the last firing. A chunk is `steps_between_updates` wide and
        rarely divides `epoch_steps`, so counting up would overshoot by most of
        a chunk every time and the overshoot would accumulate: at a 2048-step
        chunk and a 300k epoch that is ~1k of drift per epoch, which is a whole
        lost epoch over a 3M-step run. Bucket math cannot drift — the boundary
        is a property of `steps`, not of the history of how it got there.

        It also means a jump rather than an increment (a resume, the warmup
        fill) needs no re-phasing: it simply lands wherever it lands on the grid.
        """
        return prev_steps // stride < steps // stride

    def _init_counters(self):
        """Seed the run counters from the resume metadata (all zeros for a fresh
        run), so `steps` keeps meaning total env steps across a restart."""
        self.steps = self.initial_steps
        return (self.initial_epochs, self.initial_episodes,
                self.initial_gradient_steps)

    @staticmethod
    def _precompile_update(agent, agent_key):
        """Compile the gradient step once the buffer holds warmup data. Returns
        the advanced key. No-op for agents without the update contract."""

        if not isinstance(agent.hp, OffPolicyHyperparams):
            return agent_key
        print("Compiling gradient step...", flush=True)
        t0 = time.time()
        agent_key, warm_key = jax.random.split(agent_key)
        agent.learn(warm_key)
        jax.block_until_ready(jax.tree.leaves(nnx.state(agent.state)))
        print(f"  {time.time() - t0:.1f}s", flush=True)
        return agent_key

    @staticmethod
    def _collect_steps(agent, num_envs):
        """Env steps per `rollout.collect` call — the loop's unit of work.

        ONE `steps_between_updates` window, so the chunk ends exactly where the
        learning pass fires and the actor is constant for its whole duration.
        That is what makes the fused acting chunk a throughput change and not an
        algorithm change: the per-step loop also left the actor untouched
        between passes, so the two produce the same schedule.

        An agent that declares no cadence gets 1. PPO does declare one, derived
        from its queue width rather than configured.
        """
        between = 0 if agent.hp is None else int(agent.hp.steps_between_updates)
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
        """Build the backend rollout, then train.

        A play-only baseline from `agents.basic` does not belong here and says
        so itself: its `learn` raises, and its acting cannot be compiled against
        a handed-in actor because the policy IS its own Python state.
        """
        rollout, rstate = build_rollout(
            self.environment, self.test_environment, NUM_ENVS, rngs.envs(),
        )
        try:
            self._run(rollout, rstate, rngs)
        finally:
            if self._checkpoint_manager is not None:
                self._checkpoint_manager.close()

    def _run(self, rollout, rstate: RolloutState, rngs):
        start_time = last_epoch_time = time.time()
        agent = self.agent
        num_envs = rollout.num_envs
        self.num_envs = num_envs
        agent_key = rngs.agent()

        memory_warmup = getattr(agent, "memory_warmup", 0)
        warmup_iters = self._warmup_iters(memory_warmup, num_envs)
        epochs, episodes, tot_gradient_steps = self._init_counters()

        if warmup_iters > 0:
            print(f"Warmup: {warmup_iters} iters ({memory_warmup:,} steps)...",
                  flush=True)
            rstate, warmup_episodes = rollout.warmup(
                agent, rstate, warmup_iters,
            )
            # Warmup transitions are real env steps, so they count.
            self.steps = self.initial_steps + warmup_iters * num_envs
            episodes += warmup_episodes
            print(f"  Done: {self.steps:,} steps, {int(episodes)} episodes",
                  flush=True)

        agent_key = self._precompile_update(agent, agent_key)

        print("Training...", flush=True)
        collect_steps = self._collect_steps(agent, num_envs)
        chunk_env_steps = collect_steps * num_envs
        acc = _new_epoch_acc(rollout.metric_keys)
        grad_steps = 0
        first_chunk, bench_t0, bench_steps = True, None, 0

        while True:
            chunk_t0 = time.time()
            agent.state, noise_module, rstate, sums = rollout.collect(
                agent.state, getattr(agent, "noise_module", None), rstate,
                agent.freeze_acting_norm(),
                agent=agent, n_steps=collect_steps,
            )
            if noise_module is not None:
                agent.noise_module = noise_module

            prev_steps = self.steps
            self.steps += chunk_env_steps
            bench_steps += chunk_env_steps
            acc["sums"] = acc["sums"] + sums
            acc["steps"] += chunk_env_steps
            episodes = episodes + sums.count

            agent_key, update_key = jax.random.split(agent_key)
            if agent.due_for_update(self.steps):
                gradient_steps, actor_loss, critic_loss = agent.learn(update_key)
                grad_steps += gradient_steps
                tot_gradient_steps = self.initial_gradient_steps + grad_steps
                acc["actor_losses"].append(actor_loss)
                acc["critic_losses"].append(critic_loss)

            if first_chunk:
                first_chunk = False
                jax.block_until_ready(rstate.obs)
                print(f"  First chunk: {time.time() - chunk_t0:.1f}s "
                      f"(includes the compile)", flush=True)
                bench_t0, bench_steps = time.time(), 0
            elif bench_t0 is not None and bench_steps >= num_envs * 20:
                self._bench(bench_steps, bench_t0, rstate.obs)
                bench_t0 = None

            if self.show_progress:
                logger.show_progress(self.steps, self.epoch_steps, self.max_steps)

            if self._crossed(prev_steps, self.steps, self.epoch_steps):
                epochs, acc = self._end_of_epoch(
                    agent, acc, rollout, rstate,
                    epochs=epochs, episodes=episodes,
                    gradient_steps=tot_gradient_steps,
                    start_time=start_time, last_epoch_time=last_epoch_time,
                )
                rstate, invalidated = rollout.epoch_refresh(rstate)
                if invalidated:
                    rstate = rollout.reset_tally(rstate)
                last_epoch_time = time.time()

            stop_training = self._checkpoint_if_due(
                agent, epochs=epochs, episodes=episodes,
                gradient_steps=tot_gradient_steps, prev_steps=prev_steps,
            )
            if stop_training:
                break

    def _end_of_epoch(self, agent, acc, rollout, rstate, *,
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
            pin = agent.freeze_acting_norm()
            self._store_test_metrics(*rollout.evaluate(
                agent.state.actor,
                agent.state.obs_stats if pin is None else pin,
                rollout.reset_test(), agent=agent,
            ))

        epochs += 1
        self._store_epoch_metrics(
            agent,
            self._epoch_stats(acc["sums"], rstate.scores, rstate.lengths), acc,
            epochs=epochs, episodes=episodes,
            sps=self.steps / (time.time() - start_time),
            gradient_steps=gradient_steps,
            times=(start_time, last_epoch_time),
        )
        logger.dump(step=self.steps)
        return epochs, _new_epoch_acc(rollout.metric_keys)

    def _checkpoint_if_due(self, agent, *, epochs, episodes,
                           gradient_steps, prev_steps):
        """Saves on the save cadence and at the step budget.

        Returns:
            Whether the step budget is spent and training should stop.
        """
        stop_training = self.steps >= self.max_steps
        if stop_training or self._crossed(prev_steps, self.steps,
                                          self.save_steps):
            self._save(agent, epochs, episodes, gradient_steps)
        return stop_training

    @staticmethod
    def _epoch_stats(sums, scores, lengths):
        """(n, mean/std of return, mean/std of length) over episodes that
        completed this epoch, falling back to the in-flight per-env counters when
        none did (episodes longer than an epoch) so the line is never blank."""
        n = int(sums.count)
        scores, lengths = np.asarray(scores), np.asarray(lengths)
        if n == 0:
            return (n, float(np.mean(scores)), float(np.std(scores)),
                    float(np.mean(lengths)), float(np.std(lengths)))

        mean_ret = float(sums.ret / sums.count)
        mean_len = float(sums.length / sums.count)

        # Clamped against tiny negative variances from float round-off.
        def std(total_sq, mean):
            return float(np.sqrt(max(float(total_sq / sums.count) - mean ** 2, 0.0)))

        return (n, mean_ret, std(sums.ret_sq, mean_ret),
                mean_len, std(sums.length_sq, mean_len))

    def _store_epoch_metrics(
        self, agent, stats, acc, *, epochs, episodes, sps, gradient_steps,
        times,
    ):
        """Store one epoch's metrics under the shared namespace scheme."""
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

        for name in ("actor", "critic"):
            values = acc[f"{name}_losses"]
            logger.store(f"train/loss/{name}",
                         float(np.mean(values)) if values else None)


        iters = acc["steps"] / self.num_envs
        sums = acc["sums"]
        extra = {k: float(v / iters) for k, v in sums.metrics.items()} if iters else {}
        if iters:
            extra["noise/per_joint_abs"] = float(sums.noise / iters)
        # Every agent answers this; the ones with nothing to report answer {}.
        extra.update(agent.pop_diagnostics(self.steps))
        for k, v in extra.items():
            logger.store(f"train/{k}", float(v))

        now = time.time()
        logger.store("sys/sps", sps)
        logger.store("sys/time/total_s", now - start_time)
        logger.store("sys/time/epoch_s", now - last_epoch_time)
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
        stats land in the same dump as the training stats.

        The eval is one compiled program, so its results arrive on device; this
        is the one sync per epoch that brings them back.
        """
        scores, lengths = np.asarray(scores), np.asarray(lengths)
        start_obs = np.asarray(start_obs)
        logger.store("test/score", float(np.mean(scores)))
        logger.store("test/score/std", float(np.std(scores)))
        logger.store("test/length", float(np.mean(lengths)))
        logger.store("test/length/std", float(np.std(lengths)))
        logger.store("test/score_per_step",
                     float(np.mean(scores / np.maximum(lengths, 1))))
        logger.store("test/distinct_starts",
                     float(len(np.unique(start_obs, axis=0))))
