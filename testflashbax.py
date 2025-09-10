# pip install flashbax jax jaxlib

import jax
import jax.numpy as jnp
import flashbax as fbx

# -----------------------------
# Config
# -----------------------------
n_envs = 400
obs_dim = 8
act_dim = 2
sample_batch_size = 256

# -----------------------------
# Build a Flat Buffer for batched adds (no time dim in add(...))
# -----------------------------
buffer = fbx.make_flat_buffer(
    max_length=100_000,
    min_length=3,               # need >= 3 timesteps to have 2 transitions
    sample_batch_size=sample_batch_size,
    add_sequences=False,        # <-- we add single transitions per call
    add_batch_size=n_envs,      # <-- tells Flashbax your add batch size
)

# -----------------------------
# Init with a SINGLE TIMESTEP (no batch/time dims)
# -----------------------------
example_timestep = {
    "obs":      jnp.zeros((obs_dim,)),
    "action":   jnp.zeros((act_dim,)),
    "reward":   jnp.array(0.0),
    "discount": jnp.array(1.0),    # use 0.0 on terminal
}
state = buffer.init(example_timestep)

# -----------------------------
# Toy data generator for one step across all 400 envs
# IMPORTANT: shapes are (B, …)  -> NO time axis here
# -----------------------------
def make_step(rng):
    k1, k2, k3 = jax.random.split(rng, 3)
    obs      = jax.random.normal(k1, (n_envs, obs_dim))                 # (400, 8)
    action   = jax.random.uniform(k2, (n_envs, act_dim), minval=-1.0, maxval=1.0)     # (400, 2)
    reward   = jax.random.normal(k3, (n_envs,))                         # (400,)
    discount = jnp.ones((n_envs,))                                      # (400,)
    return {"obs": obs, "action": action, "reward": reward, "discount": discount}

# -----------------------------
# Add 10 sequential timesteps (internally Flashbax treats them as T=1 each call)
# -----------------------------
key = jax.random.PRNGKey(0)
for _ in range(10):
    key, sub = jax.random.split(key)
    step = make_step(sub)
    # sanity-check shapes; none should have a time axis
    assert step["obs"].shape == (n_envs, obs_dim)
    assert step["action"].shape == (n_envs, act_dim)
    assert step["reward"].shape == (n_envs,)
    assert step["discount"].shape == (n_envs,)
    state = buffer.add(state, step)

# -----------------------------
# Sample a batch of transitions (t, t+1)
# -----------------------------
assert buffer.can_sample(state), "Buffer cannot sample yet—need more timesteps."

key, sub = jax.random.split(key)
batch = buffer.sample(state, sub)

print(type(batch))  # flashbax.buffers.flat_buffer.TransitionSample
print(type(batch.experience)) # flashbax.buffers.flat_buffer.ExperiencePair
print(type(batch.experience.first)) # dict

print("batch keys:", batch.experience.first.keys()) # experience pair of (t, t+1)
print("batch keys second:", batch.experience.second.keys())

print("obs_t    :", batch.experience.first["obs"].shape)     # (256, 8)
print("obs_t+1  :", batch.experience.second["obs"].shape)    # (256, 8)
print("action_t :", batch.experience.first["action"].shape)  # (256, 2)
print("reward_t :", batch.experience.first["reward"].shape)  # (256,)
