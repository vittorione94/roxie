from mujoco_playground import registry
env = registry.load('CartpoleBalance')

# parallel_simulation.py

import jax
import jax.numpy as jnp
import flax.struct  # For creating JAX-native data structures
from mujoco_playground import registry
from typing import NamedTuple

# --- Simulation Parameters ---
# The number of environments to run in parallel on the GPU.
NUM_ENVS = 4096
# The number of simulation steps to run in each environment.
EPISODE_LENGTH = 1000
# The name of the environment to load from the playground registry.
ENV_NAME = 'CartpoleBalance'

# --- JAX Setup ---
# Set up the master pseudo-random number generator key.
# All random operations in JAX are derived from this key.
key = jax.random.PRNGKey(seed=0)

# --- Environment Loading ---
# Load the base environment. This instance is stateless and will be used
# as a template for the vectorized functions.
env = registry.load(ENV_NAME)
print(f"Loaded environment: {ENV_NAME}")
print(f"Action size: {env.action_size}")
print(f"Observation size: {env.observation_size}")

# --- Vectorization and JIT Compilation ---

# Create a vectorized version of the environment's reset function.
# jax.vmap will map the reset function over the first axis of its input (a batch of keys).
v_reset = jax.vmap(env.reset)

# Create a vectorized version of the step function.
# jax.vmap will map over the first axis of both the states and actions.
v_step = jax.vmap(env.step)

# Apply JIT compilation to the vectorized functions for maximum performance.
# This compiles the entire batched operation into a single optimized kernel.
jit_v_reset = jax.jit(v_reset)
jit_v_step = jax.jit(v_step)

print("Successfully created JIT-compiled, vectorized reset and step functions.")

# --- On-Device Replay Buffer Definition ---

# A simple dataclass to hold one transition (s, a, r, s', d).
# Using flax.struct.dataclass makes it a JAX PyTree.
@flax.struct.dataclass
class Transition:
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_observation: jnp.ndarray
    done: jnp.ndarray

# The main replay buffer structure. It holds arrays for all transitions.
# The leading dimensions will be (NUM_ENVS, EPISODE_LENGTH).
class ReplayBuffer(NamedTuple):
    data: Transition
    # A pointer to the current index for insertion.
    current_index: int
    # The current size of the buffer.
    size: int


"""
A table specifying the structure and dimensions of the replay buffer data.
This provides a clear blueprint for the data layout.

| Buffer Component | Data Type | Shape | Description |
|--------------------|---------------|-----------------------------------------|--------------------------------------------------------------|
| observation | jnp.float32 | (NUM_ENVS, EPISODE_LENGTH, obs_dim) | Stores the observation from each environment at each timestep. |
| action | jnp.float32 | (NUM_ENVS, EPISODE_LENGTH, action_dim) | Stores the action taken in each environment at each timestep.|
| reward | jnp.float32 | (NUM_ENVS, EPISODE_LENGTH) | Stores the scalar reward received. |
| next_observation | jnp.float32 | (NUM_ENVS, EPISODE_LENGTH, obs_dim) | Stores the resulting observation after the action was taken. |
| done | jnp.float32 | (NUM_ENVS, EPISODE_LENGTH) | Float flag indicating if the episode terminated (0.0 or 1.0).|
"""
def init_replay_buffer(num_envs, episode_length, obs_dim, action_dim) -> ReplayBuffer:
    """Initializes the replay buffer with zeros."""
    # Pre-allocate memory on the device.
    buffer_shape = (num_envs, episode_length)
    transition_data = Transition(
        observation=jnp.zeros(buffer_shape + (obs_dim,), dtype=jnp.float32),
        action=jnp.zeros(buffer_shape + (action_dim,), dtype=jnp.float32),
        reward=jnp.zeros(buffer_shape, dtype=jnp.float32),
        next_observation=jnp.zeros(buffer_shape + (obs_dim,), dtype=jnp.float32),
        # FIX: The 'done' buffer now correctly uses float32 to match the environment's output.
        done=jnp.zeros(buffer_shape, dtype=jnp.float32),
    )
    return ReplayBuffer(data=transition_data, current_index=0, size=0)

@jax.jit
def add_to_replay_buffer(buffer: ReplayBuffer, transition: Transition) -> ReplayBuffer:
    """Adds a batch of transitions to the replay buffer."""
    # Get the current index for insertion.
    idx = buffer.current_index
    
    # Use jax.lax.dynamic_update_slice to insert the new data.
    # This is the JAX-native way to update parts of an array.
    new_data = jax.tree_util.tree_map(
        lambda buffer_leaf, transition_leaf: jax.lax.dynamic_update_slice_in_dim(
            buffer_leaf, jnp.expand_dims(transition_leaf, axis=1), idx, axis=1
        ),
        buffer.data,
        transition,
    )

    # Update the index and size, wrapping around if the buffer is full.
    new_index = (idx + 1) % EPISODE_LENGTH
    new_size = jnp.minimum(buffer.size + 1, EPISODE_LENGTH)
    
    return buffer._replace(data=new_data, current_index=new_index, size=new_size)

# --- Main Simulation Loop ---

# 1. Initialize the environments
print("Initializing environments...")
# Split the master key to get a unique key for each parallel environment.
key, reset_key = jax.random.split(key)
reset_keys = jax.random.split(reset_key, NUM_ENVS)
# Call the vectorized reset function to get the initial states for all envs.
states = jit_v_reset(reset_keys)

# 2. Initialize the replay buffer
print("Initializing replay buffer...")
replay_buffer = init_replay_buffer(
    NUM_ENVS, EPISODE_LENGTH, env.observation_size, env.action_size
)

# 3. Run the simulation loop
print(f"Running parallel simulation for {EPISODE_LENGTH} steps...")
for i in range(EPISODE_LENGTH):
    # Split the key for this timestep's random actions.
    key, action_key = jax.random.split(key)
    
    # Generate a batch of random actions, one for each environment.
    # The action space is assumed to be in the range [-1, 1].
    actions = jax.random.uniform(
        action_key,
        shape=(NUM_ENVS, env.action_size),
        minval=-1.0,
        maxval=1.0,
    )
    
    # Store the current observation before stepping.
    current_obs = states.obs
    
    # Step all environments in parallel.
    next_states = jit_v_step(states, actions)
    
    # Create a transition object from the collected data.
    transition = Transition(
        observation=current_obs,
        action=actions,
        reward=next_states.reward,
        next_observation=next_states.obs,
        done=next_states.done,
    )
    
    # Add the batch of transitions to the replay buffer.
    replay_buffer = add_to_replay_buffer(replay_buffer, transition)
    
    # Update the states for the next iteration.
    states = next_states

    if (i + 1) % 100 == 0:
        print(f"  Step {i+1}/{EPISODE_LENGTH} completed.")

# 4. Final check
# The.block_until_ready() ensures all GPU computations are finished
# before we print the final message.
replay_buffer.size.block_until_ready()
print("\nSimulation finished.")
print(f"Replay buffer filled with {replay_buffer.size * NUM_ENVS} total transitions.")
print(f"Final buffer index: {replay_buffer.current_index}")