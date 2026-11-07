"""Multi-task training: independent tasks as one batch, end to end.

What has to hold for one policy to train on several tasks at once, pinned
against toy envs on the JAX path and two small tasks on `mujoco.batch`:

* each task's worlds see exactly what that task's own env produces, padded to
  the widest task and tagged with the task's one-hot;
* the observation statistics keep one row per task, and a padded column
  normalizes to exactly zero;
* the replay can be weighted by task, and moving the weights between learning
  passes recompiles nothing;
* the epoch reports a score per task, for training and for eval.
"""

from pathlib import Path

import flashbax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from roxie.agents.utils import Transition, build_agent, build_replay
from roxie.environment import functional
from roxie.environment.loader import (
    EnvBundle,
    build_playground_env,
    requested_impl,
    uses_batch,
)
from roxie.environment.multitask import (
    MultiTaskEnv,
    TaskLayout,
    allocate_worlds,
    build_multitask_env,
)
from roxie.environment.vector import JaxVectorEnv
from roxie.utils.curriculum import LearningProgressTasks, WeightedTasks
from roxie.utils.math import (
    init_obs_stats,
    normalize_obs,
    obs_mean_std,
    update_obs_stats,
)
from roxie.utils.rollout import ChunkSums, build_rollout
from roxie.utils.trainer import Trainer
from tests.test_env_protocol import CounterEnv

REPO = Path(__file__).resolve().parent.parent


class WideCounter(CounterEnv):
    """`CounterEnv` with extra observation columns, so two tasks differ in width."""

    def __init__(self, width: int, start: float = 0.0):
        super().__init__(start)
        self.width = width
        self.observation_space = functional.unbounded_box(width)

    def observation(self, state, rng, params=None):
        base = jnp.stack([state.x, state.origin])
        return jnp.concatenate(
            [base, jnp.full((self.width - 2,), 3.0, dtype=jnp.float32) + state.x]
        )


def build_counter_env(*, width=2, start=0.0, actions=1, mode="train",
                      num_envs=1, test_episodes=1, seed=0):
    """A toy builder with the injected keywords every builder takes."""
    env = WideCounter(width, start)
    env.action_space = functional.box(-1.0, 1.0, shape=(actions,))
    return EnvBundle(
        env=JaxVectorEnv(env, num_envs, max_episode_steps=3),
        test_env=JaxVectorEnv(env, test_episodes, max_episode_steps=3),
        env_cfg=None,
    )


COUNTER = f"{__name__}.build_counter_env"
TOY_TASKS = {"narrow": {"width": 2}, "wide": {"width": 4, "start": 1.0}}


def _toy(num_envs=4, weights=None, test_episodes=1):
    return build_multitask_env(
        TOY_TASKS, task_defaults={"_target_": COUNTER}, weights=weights,
        num_envs=num_envs, test_episodes=test_episodes,
    )


# ---- the world split -------------------------------------------------------


def test_worlds_split_in_proportion_and_sum_exactly():
    assert allocate_worlds([1, 1, 2, 4], 16) == (2, 2, 4, 8)
    assert sum(allocate_worlds([1, 3, 7], 1000)) == 1000
    # Every task keeps a world, however small its weight.
    assert min(allocate_worlds([1, 1000], 4)) >= 1


def test_world_split_refuses_what_it_cannot_honour():
    with pytest.raises(ValueError, match="positive"):
        allocate_worlds([1, 0], 4)
    with pytest.raises(ValueError, match="cannot cover"):
        allocate_worlds([1, 1, 1], 2)


def test_layout_pads_to_the_widest_task_and_appends_the_one_hot():
    layout = TaskLayout(("a", "b"), (1, 2), (2, 3))
    obs = layout.compose([np.ones((1, 2), np.float32),
                          np.full((2, 3), 2.0, np.float32)])
    assert obs.shape == (3, 3 + 2)
    np.testing.assert_array_equal(obs[0], [1, 1, 0, 1, 0])
    np.testing.assert_array_equal(obs[1:], [[2, 2, 2, 0, 1]] * 2)
    np.testing.assert_array_equal(layout.task_ids, [0, 1, 1])


# ---- the JAX composite -----------------------------------------------------


def test_jax_composite_steps_each_task_as_its_own_env(rng_key):
    """Every block of worlds is exactly its own task's driver, padded."""
    env = _toy().env
    assert isinstance(env, MultiTaskEnv)
    assert env.single_observation_space.shape == (4 + 2,)
    state, _ = env.reset(rng_key)
    pool, _ = env.reset(jax.random.fold_in(rng_key, 1))
    action = jnp.full((4, 1), 0.5, dtype=jnp.float32)

    next_state, ts = env.step(state, action, rng_key, pool)
    keys = jax.random.split(rng_key, 2)
    for k, sub in enumerate(env.envs):
        sl = env.layout.slices[k]
        _, own = sub.step(state.env_state[k], action[sl], keys[k],
                          pool.env_state[k])
        width = own.obs.shape[1]
        np.testing.assert_allclose(ts.obs[sl, :width], own.obs)
        np.testing.assert_array_equal(ts.obs[sl, width:4], 0.0)
        np.testing.assert_array_equal(
            ts.obs[sl, 4:], np.broadcast_to(np.eye(2)[k], ts.obs[sl, 4:].shape))
        np.testing.assert_allclose(ts.reward[sl], own.reward)
    assert set(ts.info["metrics"]) >= {"nonfinite", "task/narrow/reward",
                                       "task/wide/reward", "task/wide/x"}
    assert ts.info["metrics"]["task/wide/x"].shape == (1,)


def test_tasks_must_share_an_action_space():
    with pytest.raises(ValueError, match="action space"):
        build_multitask_env(
            {"one": {"actions": 1}, "two": {"actions": 2}},
            task_defaults={"_target_": COUNTER}, num_envs=2,
        )


def test_weights_must_name_known_tasks():
    with pytest.raises(ValueError, match="unknown"):
        _toy(weights={"narrow": 1, "nope": 2})


def test_eval_gets_the_same_worlds_per_task_whatever_the_weights():
    bundle = _toy(num_envs=8, weights={"narrow": 1, "wide": 3}, test_episodes=3)
    assert bundle.env.task_counts == (2, 6)
    assert bundle.test_env.task_counts == (3, 3)


# ---- tasks on mujoco.batch --------------------------------------------------

BATCH_TASKS = {"cartpole": {"env_name": "CartpoleBalance"},
               "pendulum": {"env_name": "PendulumSwingup"}}
BATCH_DEFAULTS = {"_target_": "roxie.environment.loader.build_playground_env",
                  "impl": "batch", "num_threads": 1}


def test_batch_composite_matches_standalone_tasks(rng_key):
    """Same starts, same actions: each block is exactly its own task, padded."""
    env = build_multitask_env(BATCH_TASKS, task_defaults=BATCH_DEFAULTS,
                              num_envs=4).env
    assert isinstance(env, MultiTaskEnv)
    alone = [build_playground_env(block["env_name"], impl="batch", num_envs=2,
                                  num_threads=1).env
             for block in BATCH_TASKS.values()]

    state, _ = env.reset(rng_key)
    keys = jax.random.split(rng_key, 2)
    own = [task.reset(k)[0] for task, k in zip(alone, keys)]
    width = env.layout.width
    rng = np.random.default_rng(0)
    for _ in range(5):
        action = jnp.asarray(rng.uniform(-1, 1, size=(4, 1)), jnp.float32)
        state, ts = env.step(state, action, rng_key)
        for k, task in enumerate(alone):
            sl = env.layout.slices[k]
            own[k], step = task.step(own[k], action[sl], rng_key)
            obs = np.asarray(ts.obs[sl])
            np.testing.assert_allclose(ts.reward[sl], step.reward)
            np.testing.assert_allclose(obs[:, :step.obs.shape[1]], step.obs)
            np.testing.assert_array_equal(obs[:, step.obs.shape[1]:width], 0.0)
            np.testing.assert_array_equal(
                obs[:, width:], np.broadcast_to(np.eye(2)[k], obs[:, width:].shape))


def test_batch_composite_runs_through_the_rollout(rng_key):
    bundle = build_multitask_env(BATCH_TASKS, task_defaults=BATCH_DEFAULTS,
                                 num_envs=4, test_episodes=2)
    rollout, rstate = build_rollout(bundle.env, bundle.test_env, 4, rng_key)
    assert rollout.reset_test().obs.shape == (4, bundle.env.layout.width + 2)
    _, _, ts = rollout.step(rstate, jnp.zeros((4, 1), jnp.float32), rng_key)
    assert ts.obs.shape == (4, bundle.env.layout.width + 2)


def test_tasks_must_share_a_backend():
    with pytest.raises(ValueError, match="same backend"):
        build_multitask_env(
            {"toy": {"_target_": COUNTER},
             "native": {**BATCH_DEFAULTS, "env_name": "CartpoleBalance"}},
            num_envs=2,
        )


# ---- the config surface ----------------------------------------------------


def test_loader_reads_the_backend_through_the_tasks():
    """Where train.py decides the CPU pin and Warp's memory setup."""
    block = OmegaConf.create({
        "_target_": "roxie.environment.multitask.build_multitask_env",
        "task_defaults": BATCH_DEFAULTS,
        "tasks": BATCH_TASKS,
    })
    assert uses_batch(block)
    warp = OmegaConf.create({
        "_target_": "roxie.environment.multitask.build_multitask_env",
        "task_defaults": {"impl": "warp"},
        "tasks": {"walk": {"env_name": "WalkerWalk"},
                  "run": {"env_name": "WalkerRun"}},
    })
    assert requested_impl(warp) == "warp" and not uses_batch(warp)
    assert requested_impl(OmegaConf.create({"impl": "jax"})) == "jax"


def test_humanoid_launchable_composes_the_three_tasks_on_warp():
    from hydra import compose, initialize_config_dir

    from roxie.utils import hydra_searchpath

    hydra_searchpath.register()
    with initialize_config_dir(config_dir=str(REPO / "roxie" / "configs"),
                               version_base=None):
        cfg = compose(config_name="humanoid/sac")
    assert list(cfg.env.tasks) == ["stand", "walk", "run"]
    assert requested_impl(cfg.env) == "warp"
    assert not uses_batch(cfg.env)


# ---- per-task observation statistics ---------------------------------------


def _two_task_batch():
    rng = np.random.default_rng(1)
    layout = TaskLayout(("a", "b"), (300, 300), (3, 5))
    parts = [rng.normal(5.0, 3.0, size=(300, 3)),
             rng.normal(-2.0, 0.5, size=(300, 5))]
    return layout.compose([p.astype(np.float32) for p in parts])


def test_each_task_normalizes_by_its_own_moments():
    obs = _two_task_batch()
    stats = update_obs_stats(init_obs_stats(5, 2), obs)
    mean, std = obs_mean_std(stats, 1e-8)
    z = np.asarray(normalize_obs(obs, mean, std, None))

    a, b = z[:300], z[300:]
    np.testing.assert_allclose(a[:, :3].mean(0), 0.0, atol=1e-5)
    np.testing.assert_allclose(a[:, :3].std(0), 1.0, atol=1e-4)
    np.testing.assert_allclose(b[:, :5].std(0), 1.0, atol=1e-4)
    # The observation mask: a column task `a` does not fill is exactly zero...
    np.testing.assert_array_equal(a[:, 3:5], 0.0)
    # ...and the one-hot reaches the network untouched.
    np.testing.assert_array_equal(z[:, 5:], np.asarray(obs[:, 5:]))


def test_an_all_zero_bootstrap_observation_stays_finite():
    """A terminated n-step window bootstraps from zeros, one-hot included."""
    stats = update_obs_stats(init_obs_stats(5, 2), _two_task_batch())
    mean, std = obs_mean_std(stats, 1e-8)
    assert jnp.isfinite(normalize_obs(jnp.zeros((2, 7)), mean, std, 5.0)).all()


def test_single_task_statistics_are_the_one_row_case():
    batch = jnp.asarray(np.random.default_rng(2).normal(size=(64, 4)),
                        dtype=jnp.float32)
    stats = update_obs_stats(init_obs_stats(4), batch)
    mean, std = obs_mean_std(stats, 1e-8)
    expected = (batch - jnp.mean(batch, 0)) / jnp.sqrt(jnp.var(batch, 0) + 1e-8)
    np.testing.assert_allclose(normalize_obs(batch, mean, std, None), expected,
                               atol=1e-4)


# ---- the task-weighted replay ----------------------------------------------


def _filled_replay(n_step, counts=(2, 2, 4, 8), steps=50):
    layout = TaskLayout("abcd", counts, (3,) * len(counts))
    n, k = sum(counts), len(counts)
    memory = OmegaConf.create({
        "_target_": "flashbax.buffers.make_flat_buffer",
        # Room for a 4096-sample draw, which flashbax checks against capacity.
        "max_length": n * 512, "min_length": n, "sample_batch_size": 4096,
        "add_sequences": False, "add_batch_size": n,
    })
    prototype = Transition(
        observation=jnp.zeros(3 + k), action=jnp.zeros(1), reward=jnp.zeros(()),
        terminal=jnp.zeros((), bool), truncation=jnp.zeros((), bool),
    )
    replay = build_replay(memory, n_step, prototype, task_ids=layout.task_ids)
    state = replay.init()
    rng = np.random.default_rng(0)
    for _ in range(steps):
        x = rng.normal(size=(n, 3)).astype(np.float32)
        state = replay.add(state, Transition(
            observation=layout.compose([x[sl] for sl in layout.slices]),
            action=jnp.zeros((n, 1)), reward=jnp.zeros(n),
            terminal=jnp.zeros(n, bool), truncation=jnp.zeros(n, bool),
        ))
    return replay, state, k


@pytest.mark.parametrize("n_step", [1, 5], ids=["flat", "trajectory"])
def test_replay_draws_tasks_by_weight_not_by_world_count(n_step):
    replay, state, k = _filled_replay(n_step)
    target = jnp.asarray([0.1, 0.2, 0.3, 0.4], jnp.float32)
    batch = jax.jit(replay.draw)(state, jax.random.PRNGKey(0), target)
    head = batch.experience.first if n_step == 1 else batch.experience
    tasks = np.asarray(head.observation[..., -k:]).argmax(-1)
    if tasks.ndim == 2:
        assert np.all(tasks == tasks[:, :1]), "a window crossed rows"
        tasks = tasks[:, 0]
    np.testing.assert_allclose(np.bincount(tasks, minlength=k) / tasks.size,
                               target, atol=0.03)


def test_unweighted_draw_is_flashbax_itself():
    replay, state, _ = _filled_replay(1)
    key = jax.random.PRNGKey(3)
    np.testing.assert_array_equal(
        replay.draw(state, key).experience.first.observation,
        replay.buffer.sample(state, key).experience.first.observation,
    )


# ---- an agent on a multi-task env ------------------------------------------


def _sac_block():
    """The shipped SAC config, shrunk; its `${env.*}` interpolations intact."""
    cfg = OmegaConf.load(REPO / "roxie" / "configs" / "agent" / "sac.yaml")
    cfg.actor_config.features = [16, 16]
    cfg.critic_config.critic1.features = [16, 16]
    cfg.critic_config.critic2.features = [16, 16]
    cfg.memory_config.update(max_length=256, min_length=8, sample_batch_size=16)
    cfg.hyperparams.update(memory_warmup=16, steps_between_updates=8,
                           learning_steps=2)
    return cfg


def _sac(env):
    cfg = OmegaConf.create({
        "env": {"parallel_envs": env.num_envs, "obs_size": 6, "action_size": 1,
                "obs_action_size": 7},
        "agent": _sac_block(),
    }).agent
    return build_agent(
        cfg, env_obs_size=6, env_action_size=1,
        action_low=-jnp.ones(1, jnp.float32), action_high=jnp.ones(1, jnp.float32),
        env_task_counts=env.task_counts,
    )


def test_off_policy_agent_trains_on_the_composite_and_reweights_freely(rng_key):
    """The fused chunk, a learning pass, and a curriculum's reweighting."""
    from roxie.agents import sac

    bundle = _toy()
    agent = _sac(bundle.env)
    assert agent.state.obs_stats.sum.shape == (2, 4)
    np.testing.assert_allclose(agent.state.task_weights, [0.5, 0.5])

    rollout, rstate = build_rollout(bundle.env, bundle.test_env, 4, rng_key)
    assert rollout.task_names == ("narrow", "wide")
    rstate, _ = rollout.warmup(agent, rstate, 8)
    agent.state, _, rstate, sums = rollout.collect(
        agent.state, None, rstate, None, agent=agent, n_steps=6,
    )
    # Three-step episodes: both tasks finished some in a six-step chunk.
    assert sums.task_count.shape == (2,) and np.all(np.asarray(sums.task_count) > 0)

    agent.learn(jax.random.PRNGKey(1))
    compiled = sac._sac_grad_steps._cache_size()
    agent.set_task_weights([0.9, 0.1])
    agent.learn(jax.random.PRNGKey(2))
    assert sac._sac_grad_steps._cache_size() == compiled, "reweighting recompiled"

    with pytest.raises(ValueError, match="task weights"):
        agent.set_task_weights([1.0, 1.0, 1.0])
    for invalid in ([0.0, 0.0], [-0.5, 1.5], [np.nan, 1.0], [np.inf, 1.0]):
        with pytest.raises(ValueError, match="non-negative"):
            agent.set_task_weights(invalid)
    agent.set_task_weights([0.0, 1.0])


def test_on_policy_and_single_task_agents_have_no_task_weights():
    from tests import harness

    with pytest.raises(ValueError, match="task-weighted"):
        harness.build("ppo").set_task_weights([1.0])
    with pytest.raises(ValueError, match="task-weighted"):
        harness.build("sac").set_task_weights([1.0])


# ---- per-task scores --------------------------------------------------------


def test_chunk_sums_split_finished_episodes_by_task():
    hot = np.eye(2, dtype=np.float32)[[0, 0, 1, 1]]
    ep_ret = jnp.asarray([[3.0, 0.0, 5.0, 7.0]])
    done = jnp.asarray([[1.0, 0.0, 1.0, 1.0]])
    sums = ChunkSums.from_chunk(ep_ret, ep_ret, done, jnp.zeros_like(ep_ret),
                                {}, n_steps=1, task_hot=hot)
    np.testing.assert_allclose(sums.task_ret, [3.0, 12.0])
    np.testing.assert_allclose(sums.task_count, [1.0, 2.0])
    np.testing.assert_allclose(Trainer._task_scores(sums), [3.0, 6.0])


def test_a_task_with_nothing_finished_scores_nan_not_zero():
    sums = ChunkSums.zeros((), num_tasks=2)
    assert np.isnan(Trainer._task_scores(sums)).all()
    # A single-task env carries no per-task sums at all.
    assert ChunkSums.zeros(()).task_ret is None


# ---- curricula --------------------------------------------------------------


def test_weighted_curriculum_orders_weights_by_task():
    curriculum = WeightedTasks({"run": 3})
    np.testing.assert_allclose(curriculum.start(("stand", "run")), [0.25, 0.75])
    np.testing.assert_allclose(curriculum.update([np.nan, np.nan]), [0.25, 0.75])
    with pytest.raises(ValueError, match="unknown"):
        WeightedTasks({"fly": 1}).start(("stand", "run"))


def test_learning_progress_favours_the_task_that_is_moving():
    curriculum = LearningProgressTasks(score_scale=1000.0, floor=0.2)
    curriculum.start(("stand", "walk", "run"))
    for epoch in range(8):
        weights = curriculum.update([950.0, 300.0, 50.0 + 60.0 * epoch])
    assert np.argmax(weights) == 2
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert weights.min() >= 0.2 / 3 - 1e-9, "the floor starves nothing"


def test_learning_progress_starts_uniform():
    curriculum = LearningProgressTasks(score_scale=1.0)
    np.testing.assert_allclose(curriculum.start(("a", "b", "c", "d")), [0.25] * 4)


def test_the_first_learning_pass_draws_by_the_curriculum(tmp_path):
    from flax import nnx

    from roxie.utils import logger

    bundle = _toy()
    agent = _sac(bundle.env)
    seen = []
    learn = agent.learn

    def spy(key, n_steps=None):
        seen.append(np.asarray(agent.state.task_weights))
        return learn(key, n_steps)

    agent.learn = spy
    trainer = Trainer(
        output_dir=str(tmp_path), steps=64, epoch_steps=1_000, save_steps=1_000,
        show_progress=False, curriculum=WeightedTasks({"wide": 3}),
    )
    trainer.initialize(agent=agent, environment=bundle.env,
                       test_environment=bundle.test_env)
    logger.initialize(path=str(tmp_path / "logs"), backends=[])
    try:
        trainer.run(4, nnx.Rngs(envs=0, agent=3))
    finally:
        logger.close()
    np.testing.assert_allclose(seen[0], [0.25, 0.75])


def test_learning_progress_holds_a_task_that_finished_nothing():
    curriculum = LearningProgressTasks(score_scale=1.0)
    curriculum.start(("a", "b"))
    curriculum.update([0.5, 0.5])
    before = curriculum._fast.copy()
    curriculum.update([np.nan, 0.9])
    assert curriculum._fast[0] == before[0]


# ---- playback of one task ---------------------------------------------------


def _run_config(tasks=TOY_TASKS):
    return OmegaConf.create({
        "env": {
            "_target_": "roxie.environment.multitask.build_multitask_env",
            "task_defaults": {"_target_": COUNTER},
            "tasks": tasks,
            "parallel_envs": 4,
        },
        "agent": _sac_block(),
    })


def test_select_task_narrows_the_config_to_one_task():
    from roxie.utils.playback import select_task

    cfg, task = select_task(_run_config(), "wide")
    assert task == 1
    assert cfg.env._target_ == COUNTER and cfg.env.width == 4
    # Kept so the agent can rebuild the replay geometry it trained with.
    assert cfg.env.parallel_envs == 4


def test_select_task_refuses_a_task_the_run_did_not_train():
    from roxie.utils.playback import select_task

    with pytest.raises(ValueError, match="pass task_id"):
        select_task(_run_config(), None)
    with pytest.raises(ValueError, match="pass task_id"):
        select_task(_run_config(), "fly")
    with pytest.raises(ValueError, match="single task"):
        select_task(OmegaConf.create({"env": {"env_name": "WalkerWalk"}}), "walk")


def _saved_multi_task_run(tmp_path, cfg=None):
    """Saves through the trainer; returns the checkpoint's step directory."""
    bundle = _toy()
    agent = _sac(bundle.env)
    trainer = Trainer(output_dir=str(tmp_path))
    trainer.environment, trainer.steps = bundle.env, 100
    trainer._save(agent, epochs=1, episodes=0, gradient_steps=0)
    trainer._checkpoint_manager.close()
    (tmp_path / ".hydra").mkdir()
    OmegaConf.save(cfg or _run_config(), tmp_path / ".hydra" / "config.yaml")
    return str(tmp_path / "checkpoints" / "100")


def test_every_task_of_a_run_replays_under_one_agent(tmp_path):
    from roxie.utils.playback import load_tasks

    plays = load_tasks(_saved_multi_task_run(tmp_path), verbose=False)
    assert list(plays) == ["narrow", "wide"]
    assert plays["narrow"].agent is plays["wide"].agent
    assert plays["narrow"].func_env is not plays["wide"].func_env


def test_the_backend_banner_prints_once_however_many_tasks(tmp_path, capsys):
    from roxie.utils.playback import load_tasks

    load_tasks(_saved_multi_task_run(tmp_path), verbose=True)
    assert capsys.readouterr().out.count("PHYSICS BACKEND LOADED") == 1


def test_the_task_layout_reads_only_the_checkpoint_metadata(tmp_path):
    from roxie.utils.checkpoint import read_checkpoint
    from roxie.utils.playback import _task_layout

    checkpoint = _saved_multi_task_run(tmp_path)
    assert list(read_checkpoint(Path(checkpoint), only=("metadata",))) == ["metadata"]
    assert _task_layout(checkpoint) == (2, 4, (2, 4))


def test_a_task_whose_replay_width_changed_is_refused(tmp_path):
    from roxie.utils.playback import load_tasks

    cfg = _run_config()
    # The saved layout says this task trained on 4 columns.
    cfg.env.tasks.wide.width = 3
    with pytest.raises(ValueError, match="trained on 4"):
        load_tasks(_saved_multi_task_run(tmp_path, cfg), verbose=False)


def test_a_replay_whose_worlds_changed_task_is_not_restored(tmp_path):
    trainer = Trainer(output_dir=str(tmp_path), save_buffer=True)
    trainer.environment, trainer.steps = _toy().env, 100
    trainer._save(_sac(_toy().env), epochs=1, episodes=0, gradient_steps=0)
    trainer._checkpoint_manager.close()
    checkpoint = tmp_path / "checkpoints" / "100"

    assert _sac(_toy().env).restore(checkpoint)["buffer_restored"] is True
    # Same four worlds and the same buffer shapes, split (1, 3) instead.
    moved = _sac(_toy(weights={"wide": 3}).env)
    assert moved.task_counts == (1, 3)
    assert moved.restore(checkpoint)["buffer_restored"] is False


def test_a_resume_that_reorders_or_rewidens_the_tasks_is_refused(tmp_path):
    from roxie.utils.trainer import check_resumed_tasks

    checkpoint = _saved_multi_task_run(tmp_path)
    metadata = {"task_obs_widths": [2, 4]}
    check_resumed_tasks(checkpoint, _toy().env, metadata)

    reordered = build_multitask_env(
        {"wide": TOY_TASKS["wide"], "narrow": TOY_TASKS["narrow"]},
        task_defaults={"_target_": COUNTER}, num_envs=4,
    )
    with pytest.raises(ValueError, match="trained tasks"):
        check_resumed_tasks(checkpoint, reordered.env, metadata)
    with pytest.raises(ValueError, match="trained tasks"):
        check_resumed_tasks(checkpoint, _toy().env, {"task_obs_widths": [2, 3]})


def test_a_single_task_run_has_one_world_with_no_name(monkeypatch):
    from roxie.utils import playback

    monkeypatch.setattr(
        playback, "load_config",
        lambda *_: OmegaConf.create({"env": {"env_name": "WalkerWalk"}}),
    )
    monkeypatch.setattr(playback, "load_policy", lambda *_, **__: "the policy")
    assert playback.load_tasks("checkpoint", verbose=False) == {None: "the policy"}


def test_one_task_of_a_multi_task_checkpoint_plays_back(tmp_path, rng_key):
    """Saved through the trainer, replayed through the real loader."""
    from roxie.utils.playback import load_policy

    checkpoint = _saved_multi_task_run(tmp_path)
    for name, k, width in (("narrow", 0, 2), ("wide", 1, 4)):
        play = load_policy(checkpoint, task_id=name, verbose=False)
        assert play.obs_size == 4 + 2
        obs = play.observe(np.arange(width, dtype=np.float32) + 1.0)
        np.testing.assert_array_equal(obs[width:4], 0.0)
        np.testing.assert_array_equal(obs[4:], np.eye(2)[k])
        action, _, _ = play.agent.select_action(obs[None], rng_key, evaluate=True)
        assert action.shape == (1, 1)
