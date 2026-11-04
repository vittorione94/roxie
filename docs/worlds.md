# World models

What a world model is in roxie, who owns the data, and how SHAC trains through one.

*Back to the [README](../README.md).*

## The loop

Model-based training follows SuperMimic ([Liao et al., 2026](https://arxiv.org/abs/2610.28480)). Each update window runs three phases:

1. **Collect.** The policy acts in the real environments with action noise. Every transition goes into the **world's** replay buffer.
2. **Fit.** The world model fits itself on `k`-step windows of that buffer (`world.learn`).
3. **Improve.** The agent imagines `H`-step rollouts through the frozen model, starting from real observations, and backpropagates the imagined return into the actor (`agent.learn(key, world=world)`).

```
Trainer ──collect──▶ rollout.collect(agent.state, world.state, …)  ─ real transitions → world buffer
        ──if agent.due_for_update──▶ world.learn(world_key)         ─ k-step fit; donates world.state
                                  └─▶ agent.learn(key, world=world)  ─ starts sampled from the world buffer,
                                                                       H steps through world.step (frozen)
```

A run is model-based when its config composes a `world:` group ([`experiments/dmc/bench_shac.yaml`](../experiments/dmc/bench_shac.yaml)). `train.py` then builds the world next to the agent, and `Trainer.initialize` refuses a mismatched pair loudly. An agent with `uses_world = True` needs a world. A model-free agent refuses one, because nothing in it could use the world.

## A world is the agent's peer, and owns the real data

`roxie.worlds.WorldModel` sits beside `Agent` in the loop rather than inside it. It owns three things:

* **The replay buffer of real transitions.** Agents use `n_step`-style trajectory buffers; the world always uses one with `sample_sequence_length = window + 1`, built by `build_replay(memory_config, window)`. The agent then holds none: `agent.replay` and `agent.state.buffer_state` are `None`.
* **Its own observation statistics.** The model reads and predicts in a normalized space that is the world's alone, separate from the agent's acting normalizer. They fold each landed observation once. The agent's statistics keep the weighting acting has always used.
* **A pure, differentiable `step(dynamics, obs_mean, obs_std, obs, action) -> (next_obs, reward, term_prob)`.** It takes raw observations in and out, and is normalized internally *without* the clip acting applies, because a clip would zero the gradient of every state outside it.

The rollout addresses whichever object owns the buffer through the same three methods: `buffer_transitions`, `absorb_obs_stats` and `fill_buffer`. `collect_chunk` carries `world_state` in its scan next to `train_state`, writes the transition into it, and folds both normalizers after the scan. The warmup fill goes to the world's buffer, while the agent folds only the statistics of the block (`Agent.fill_buffer` skips the write when it has no replay).

Why the world owns the data: dynamics do not depend on the policy that collected them, so every transition remains valid for fitting the model for as long as the buffer keeps it. The agent only ever needs states to start imagining from, which it samples out of the world's buffer through `world.start_sampler(batch_size)`. That is a second flashbax sampler over the same state, at a sequence length of 1. Flashbax's trajectory sampling is a pure function of the state and its own sizes, so `ReplayManager.sampler` can hand out samplers of any geometry over one store.

### Donation

* `world.learn` donates `world.state`. The buffer is the largest array in the run, and not donating it means a transient second copy every pass.
* `collect_chunk` donates it too, for the same reason; a `None` world state donates nothing.
* `agent.learn` donates **only the agent's state**. The world's state is read there but still owned by the world, so donating it would free its buffer out from under the next collect chunk.

## `DeterministicWorld`

[`roxie/worlds/deterministic.py`](../roxie/worlds/deterministic.py) implements the paper's model. `DynamicsModel` ([`roxie/models/dynamics.py`](../roxie/models/dynamics.py)) is one MLP trunk over `(normalized obs, action)` with three heads: the normalized *change* in the observation, the reward, and a termination logit. Two initializations make an untrained model a neutral one:

* **The delta head starts near zero, so "nothing moves".**
* **The termination bias starts at −5, so "nothing ends".** At a zero logit every imagined step would carry a 50% chance of termination, and a 32-step rollout would keep ~1e-10 of its weight past the horizon until the termination fit caught up.

### The fit

`world_rollout_loss_fn` ([`roxie/losses/world_losses.py`](../roxie/losses/world_losses.py)) rolls the model **autoregressively** over a `k + 1`-step window: seeded at the logged first observation, then fed its own predictions and the logged actions. It is scored at every step on state (in normalized units, so every dimension counts equally), reward, and termination (BCE). Fitting one step at a time would only ever show the model true inputs. The autoregressive fit shows it its own errors, which is what lets imagination run past `k` without drifting off.

Windows are sampled without regard to episode boundaries, so the loss masks them (`window_masks`). With `done_j = terminal_j | truncation_j`:

* the reward and termination label at step `j` count if nothing **before** `j` ended;
* the observation step `j` lands on counts only if step `j` itself did not end either. The pool auto-resets, so the stored observation after a done is the first of a *new* episode.

The roxie environments expose no simulator state or domain-randomization context, so the model works in **observation space**: `s = obs`, with no `ξ` to condition on and no separate sensor-noise model.

## SHAC

[`roxie/agents/shac.py`](../roxie/agents/shac.py) is the short-horizon actor-critic of [Xu et al., 2022](https://arxiv.org/abs/2204.07137), with the learned model in place of a differentiable simulator. Each fused gradient step does four things:

1. Samples `imagined_batch_size` real observations from the world's buffer.
2. Rolls `horizon` steps through `world.step` with the deterministic actor in the graph, and takes an actor step on `−mean(J)/H`, where

   ```
   J = Σ_t γ^t c_t r_{t+1} + γ^H c_H V(s_H),   c_0 = 1,  c_{t+1} = c_t (1 − d_{t+1}).
   ```

   The survival weight `c` multiplies the model's termination *probability* in rather than cutting at a hard done, so it stays differentiable. Only the actor is differentiated. The world's network and the critic are arguments, so their parameters are inputs to the gradient, not targets of it.
3. Computes lambda-return targets on that rollout, with its gradients stopped, under the critic as it stood when the actor bootstrapped off it. This is `rlax.lambda_returns` with `γ(1 − d)` as the discount.
4. Fits the critic to those fixed targets for `critic_epochs × critic_minibatches` shuffled minibatches.

There are no target networks. Acting is DDPG's (`NoiseModuleActing`): a deterministic actor plus a noise module. The noise only widens the actions the model is fit on, so the bench uses a constant `N(0, 0.1²)` ([`experiments/dmc/noise/shac_gaussian.yaml`](../experiments/dmc/noise/shac_gaussian.yaml)). The actor's optimizer clips by global norm and then element-wise (`actor_clip_value`), as the paper does.

The world's network rides in the `fused_grad_steps` carry, unchanged across the scan, because nnx refuses a module closed over from outside the scan's trace. The buffer is plain arrays and is closed over.

## Diagnostics

* `train/loss/world`: the fit's total loss.
* `train/world/*`:
  * `state_loss`, `reward_loss`, `term_loss`;
  * `state_err_first` and `state_err_last`, the normalized state error at the window's first and last step. The last is the paper's `k`-step prediction error;
  * `valid_frac`, the share of window steps the masks kept.
* `train/shac/*`:
  * `imagined_return`, `imagined_reward`, `survival_H`;
  * `value_H`, the bootstrap at the horizon;
  * `value_target` and `value_ev`, the critic's explained variance against its targets before the fit;
  * `action_abs`.

## Checkpoints and resume

The world rides in the agent's checkpoint under a `world` key (`WorldCheckpointer`). One step directory is therefore one resumable run. `train.py` restores both halves on `resume=`, and the world's answer decides `buffer_restored`, because the buffer is the world's. A checkpoint with no `world` key, or a world trained on another observation width, is refused. Playback reads only the agent's half: acting needs no model.

## Adding a world model

Subclass `WorldModel`, set `hyperparams_cls`, build `self.state` (a `WorldState`), `self.replay` (a trajectory `ReplayManager`) and `self.diagnostics` in `__init__`, and implement `step` and `_compile_and_run`. The config's `_target_` names the class, and every sibling key is a constructor keyword, held to the same contract as an agent's yaml ([`tests/test_agent_configs.py`](../tests/test_agent_configs.py)).
