"""Minimal flashbax TrajectoryQueue walkthrough: add (B, T) batches, sample
fixed-length windows, and inspect what comes back."""

import jax
import jax.numpy as jnp
import flashbax as fbx

def tree_shapes(x):
    return jax.tree_util.tree_map(lambda a: tuple(a.shape), x)

def make_fake_batch(add_batch_size, add_sequence_length, obs_dim, offset=0.0):
    """
    Create a batch shaped (B, T, ...), matching TrajectoryQueue expectations.
    """
    B, T = add_batch_size, add_sequence_length
    obs = (jnp.arange(B * T * obs_dim, dtype=jnp.float32)
             .reshape(B, T, obs_dim)) + offset
    action = (jnp.arange(B * T, dtype=jnp.int32).reshape(B, T)) % 4
    reward = jnp.linspace(0.0, 1.0, B * T, dtype=jnp.float32).reshape(B, T)
    discount = jnp.ones((B, T), dtype=jnp.float32)
    print('fake batch shapes:', tree_shapes({"obs": obs, "action": action, "reward": reward, "discount": discount}))
    return {"obs": obs, "action": action, "reward": reward, "discount": discount}

def unwrap_experience(sample_obj):
    # Some flashbax versions wrap the result in `.experience`, others return it bare.
    return getattr(sample_obj, "experience", sample_obj)

def main():
    # ---- Config ----
    ADD_BATCH_SIZE = 2          # number of envs/streams added per 'add'
    ADD_SEQ_LEN = 4             # timesteps per add
    SAMPLE_SEQ_LEN = 3          # timesteps per sample
    MAX_LENGTH_TIME_AXIS = 10   # time capacity of the queue
    OBS_DIM = 3

    # ---- Build the queue (note required sample_sequence_length) ----
    buffer = fbx.buffers.make_trajectory_queue(
        add_batch_size=ADD_BATCH_SIZE,
        add_sequence_length=ADD_SEQ_LEN,
        sample_sequence_length=SAMPLE_SEQ_LEN,
        max_length_time_axis=MAX_LENGTH_TIME_AXIS,
    )

    # ---- Init (prototype has no leading B/T dims) ----
    example_timestep = {
        "obs": jnp.zeros((OBS_DIM,), dtype=jnp.float32),
        "action": jnp.array(0, dtype=jnp.int32),
        "reward": jnp.array(0.0, dtype=jnp.float32),
        "discount": jnp.array(1.0, dtype=jnp.float32),
    }
    state = buffer.init(example_timestep)

    print("=== After init ===")
    print("can_add?     ", bool(buffer.can_add(state)))
    print("can_sample?  ", bool(buffer.can_sample(state)))

    # ---- Add two (B,T,...) batches ----
    batch0 = make_fake_batch(ADD_BATCH_SIZE, ADD_SEQ_LEN, OBS_DIM, offset=0.0)
    state = buffer.add(state, batch0)
    print("\n=== After 1st add ===")
    print("can_add?     ", bool(buffer.can_add(state)))
    print("can_sample?  ", bool(buffer.can_sample(state)))

    batch1 = make_fake_batch(ADD_BATCH_SIZE, ADD_SEQ_LEN, OBS_DIM, offset=100.0)
    state = buffer.add(state, batch1)
    print("\n=== After 2nd add ===")
    print("can_add?     ", bool(buffer.can_add(state)))
    print("can_sample?  ", bool(buffer.can_sample(state)))

    # ---- Sample once; handle both possible return signatures ----
    try:
        state, sample = buffer.sample(state)  # common: returns (state, sample)
    except TypeError:
        sample = buffer.sample(state)         # some versions: returns sample only

    exp = unwrap_experience(sample)

    print("\n=== Sampled experience shapes ===")
    print(tree_shapes(exp))  # expect (ADD_BATCH_SIZE, SAMPLE_SEQ_LEN, ...)

    # Show a quick peek at values to verify FIFO over time
    print("\nExample obs for env 0 across sampled timesteps:")
    print(exp["obs"][0])  # shape (SAMPLE_SEQ_LEN, OBS_DIM)

    # ---- Fill beyond capacity to show FIFO behavior along time axis ----
    batch2 = make_fake_batch(ADD_BATCH_SIZE, ADD_SEQ_LEN, OBS_DIM, offset=200.0)
    state = buffer.add(state, batch2)

    try:
        state, sample2 = buffer.sample(state)
    except TypeError:
        sample2 = buffer.sample(state)
    exp2 = unwrap_experience(sample2)

    print("\n=== After overflow add + another sample ===")
    print(tree_shapes(exp2))
    print("buffer state length (time axis):", state.experience['obs'].shape)
    print("can_add?     ", bool(buffer.can_add(state)))
    print("can_sample?  ", bool(buffer.can_sample(state)))

if __name__ == "__main__":
    main()
