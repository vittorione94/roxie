"""Minimal flashbax flat-buffer walkthrough: add one batched timestep per call
(no time axis) and sample (t, t+1) transition pairs."""

import jax
import jax.numpy as jnp
import flashbax as fbx

n_envs = 400
obs_dim = 8
act_dim = 2
sample_batch_size = 256

buffer = fbx.make_flat_buffer(
    max_length=100_000,
    min_length=3,               # >= 3 timesteps to hold 2 transitions
    sample_batch_size=sample_batch_size,
    add_sequences=False,
    add_batch_size=n_envs,
)

# The prototype carries neither a batch nor a time dim.
example_timestep = {
    "obs":      jnp.zeros((obs_dim,)),
    "action":   jnp.zeros((act_dim,)),
    "reward":   jnp.array(0.0),
    "discount": jnp.array(1.0),    # 0.0 on terminal
}
state = buffer.init(example_timestep)


def make_step(rng):
    k1, k2, k3 = jax.random.split(rng, 3)
    return {
        "obs": jax.random.normal(k1, (n_envs, obs_dim)),
        "action": jax.random.uniform(
            k2, (n_envs, act_dim), minval=-1.0, maxval=1.0
        ),
        "reward": jax.random.normal(k3, (n_envs,)),
        "discount": jnp.ones((n_envs,)),
    }


key = jax.random.PRNGKey(0)
for _ in range(10):
    key, sub = jax.random.split(key)
    step = make_step(sub)
    assert step["obs"].shape == (n_envs, obs_dim)
    assert step["action"].shape == (n_envs, act_dim)
    assert step["reward"].shape == (n_envs,)
    assert step["discount"].shape == (n_envs,)
    state = buffer.add(state, step)

assert buffer.can_sample(state), "Buffer cannot sample yet—need more timesteps."

key, sub = jax.random.split(key)
batch = buffer.sample(state, sub)

print(type(batch))
print(type(batch.experience))
print(type(batch.experience.first))

print("batch keys:", batch.experience.first.keys())
print("batch keys second:", batch.experience.second.keys())

print("obs_t    :", batch.experience.first["obs"].shape)
print("obs_t+1  :", batch.experience.second["obs"].shape)
print("action_t :", batch.experience.first["action"].shape)
print("reward_t :", batch.experience.first["reward"].shape)
