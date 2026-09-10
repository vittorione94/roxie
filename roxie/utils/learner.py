"""How acting and learning are interleaved, as a strategy the loop can't see.

`Trainer._run` selects actions and hands off transitions through this surface,
so the loop body has no branch for whether learning happens inline or on a
background thread. Two implementations:

  * ``SyncLearner`` — the default. Act, buffer and grad-step on the training
    thread, in that order, every iteration.
  * ``AsyncLearner`` (in ``async_learner``) — a background thread owns
    ``agent.state`` and runs the gradient bursts while this thread keeps acting
    from a published behaviour-actor snapshot.

Shared surface:

    owns_state                        does this learner own `agent.state`?
    act(obs, key)                     -> (actions, last_noise)
    buffer(prev_obs, timestep, actions)
    update(steps)                     run the gated gradient burst, if due
    drain()                           -> (gradient_steps_since_start, [(a, c)])
    pause() / resume()                quiesce for eval and checkpointing

`owns_state` is what tells a rollout whether it may bypass `act`/`buffer` and
compile them itself: only the async learner owns `agent.state`, and there the
two calls are the hand-off (behaviour snapshot out, transitions into a queue)
rather than plain function calls.

`buffer` and `update` are separate because the rollout calls them at different
granularities: buffering once per env step, the gradient burst once per
`steps_between_updates` window. The fused JAX rollout buffers on-device inside
its scan and never calls `buffer` at all.

``drain`` reports gradient steps *since this run started*, not including any
restored from a resume — the trainer adds that back, so a resumed run's
`train/gradient_steps` continues one curve regardless of which learner is in
use.
"""

import jax
from flax import nnx

from roxie.agents.agent import Agent
from roxie.agents.utils import Transition


class SyncLearner:
    """Act, buffer and grad-step inline — the loop's default."""

    # `agent.state` stays on the acting thread, so the rollout may compile
    # acting and buffering against it directly. See `rollout.fusable`.
    owns_state = False

    def __init__(self, agent, agent_key):
        self._agent = agent
        self._key = agent_key
        self._grad_steps = 0
        self._losses = []

    def act(self, obs, key):
        actions = self._agent.step(obs, evaluate=False, key=key)
        return actions, getattr(self._agent, "last_noise", None)

    def buffer(self, prev_obs, timestep, actions):
        # Already on `agent.last_action`; taken only to share the async
        # learner's signature, where it travels by queue instead.
        del actions
        agent = self._agent
        agent.buffer_transitions(
            Transition(
                observation=prev_obs,
                action=agent.last_action,
                reward=timestep.reward,
                terminal=timestep.terminated,
                truncation=timestep.truncated,
                **(agent.last_extras or {}),
            ),
            timestep.obs,
        )

        # On top of the update `buffer_transitions` already performs.
        # Redundant-looking, but dropping it reweights the statistics away from
        # every run logged so far.
        if getattr(agent, "normalize_observations", False):
            agent.state.obs_stats = Agent.update_obs_stats(
                agent.state.obs_stats, timestep.obs,
            )

    def update(self, steps):
        # The agent gates its own updates: a Python branch on a device array
        # here would sync the host every iteration.
        self._key, update_key = jax.random.split(self._key)
        gradient_steps, actor_loss, critic_loss = self._agent.update(
            steps=steps, agent_rng=update_key,
        )
        if gradient_steps > 0:
            self._grad_steps += gradient_steps
            self._losses.append((actor_loss, critic_loss))

    def drain(self):
        losses, self._losses = self._losses, []
        return self._grad_steps, losses

    def pause(self):
        """Nothing runs off-thread, so `agent.state` is always quiescent."""

    def resume(self):
        pass

    def stop(self):
        pass


def build_learner(trainer, rollout, agent, agent_key, state):
    """Pick the learner for this run.

    Async needs three things at once: the trainer configured for it, an agent
    exposing the unconditional ``learn`` burst, and a rollout whose acting is off
    the learner's device — otherwise the "background" thread just contends for
    the GPU the env step is already saturating. Anything missing falls back to
    the synchronous learner.
    """
    if not (trainer.async_learner and hasattr(agent, "learn")):
        return SyncLearner(agent, agent_key)
    if not rollout.supports_async:
        print("async_learner requested but this backend acts on-device; "
              "falling back to the synchronous learner.", flush=True)
        return SyncLearner(agent, agent_key)

    from roxie.utils.async_learner import AsyncLearner

    return AsyncLearner.started(
        agent, agent_key, state, initial_steps=trainer.steps,
        chunk=trainer.learner_chunk,
    )
