import jax
import jax.numpy as jnp
import numpy as np

class ReplayBuffer:
    """A simple NumPy-based replay buffer."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity

        # Use NumPy for efficient in-place modification of the buffer's data
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.bool_)

        self.pointer = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """Adds a new transition to the buffer."""
        self.observations[self.pointer] = obs
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_observations[self.pointer] = next_obs
        self.terminals[self.pointer] = done

        # Increment the pointer and size, wrapping around when capacity is reached
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, jnp.ndarray]:
        """Samples a batch of transitions and converts them to JAX arrays."""
        # Generate random indices for sampling
        indices = np.random.randint(0, self.size, size=batch_size)

        # Sample a batch of data using the indices
        batch = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "terminals": self.terminals[indices],
        }

        # Convert the NumPy arrays in the batch to JAX arrays
        # This step is where data is typically moved to the accelerator (e.g., GPU/TPU)
        return jax.tree_util.tree_map(jnp.asarray, batch)
    
    def flush(self):
        """
        Resets the buffer to an empty state. 🗑️
        
        This is useful in settings like meta-RL or multi-task RL where
        you want to clear all experience between tasks.
        """
        self.pointer = 0
        self.size = 0