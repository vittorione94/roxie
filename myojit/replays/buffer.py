import jax
import jax.numpy as jnp
import flax.struct as struct
from typing import Dict
import functools

# A PyTree representing a single data transition (s, a, r, s', d).
# This structure is JAX-native and can be passed into JIT-compiled functions.
@struct.dataclass
class Transition:
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_observation: jnp.ndarray
    terminal: jnp.ndarray

# A PyTree that holds the entire state of the replay buffer.
# This includes the stored data and the buffer's internal pointers.
# All arrays are JAX arrays, ensuring they reside on the GPU.
@struct.dataclass
class BufferState:
    data: Transition
    pointer: jnp.ndarray
    size: jnp.ndarray



class JaxReplayBuffer:
    """A JAX-native, GPU-optimized replay buffer."""

    def __init__(self, capacity: int, batch_size: int):
        """
        Initializes the replay buffer with a fixed capacity.
        Note: This class is stateless. It only holds static configuration.
        """
        self.capacity = capacity
        self.batch_size = batch_size


    def init(self, transition_prototype: Transition) -> BufferState:
        """
        Initializes the buffer's state.

        Args:
            transition_prototype: A sample Transition object with the correct
                                  shapes and dtypes for pre-allocation.

        Returns:
            An initial BufferState with zeroed-out data arrays.
        """
        # Use jax.tree_util.tree_map to create zero-filled arrays that match
        # the structure and dtypes of the prototype, but with an added
        # leading dimension for the buffer's capacity.
        # https://kolonist26-jax-kr.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
        data = jax.tree_util.tree_map(
            lambda x: jnp.zeros((self.capacity, *x.shape), dtype=x.dtype),
            transition_prototype
        )
        return BufferState(
            data=data,
            pointer=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def add(self, state: BufferState, transition: Transition) -> BufferState:
        """
        Adds a new transition to the buffer. This is a pure function.

        Args:
            state: The current BufferState.
            transition: The new Transition to add.

        Returns:
            A new BufferState with the transition added.
        """
        # Use the `.at.set()` syntax for a functional update. This is the
        # recommended JAX approach for updating array elements.
        updated_data = jax.tree_util.tree_map(
            lambda buffer_leaf, transition_leaf: buffer_leaf.at[state.pointer].set(transition_leaf),
            state.data,
            transition
        )

        # Increment pointer and size, wrapping around when capacity is reached.
        new_pointer = (state.pointer + 1) % self.capacity
        new_size = jnp.minimum(state.size + 1, self.capacity)

        # Return a new state object with the updated fields.
        return state.replace(
            data=updated_data,
            pointer=new_pointer,
            size=new_size
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample(self, state: BufferState, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        Samples a batch of transitions and returns them as JAX arrays.

        Args:
            state: The current BufferState.
            batch_size: The number of transitions to sample.
            key: A jax.random.PRNGKey for random index generation.

        Returns:
            A dictionary containing the batch of sampled data.
        """
        # Generate random indices on-device using jax.random.
        indices = jax.random.randint(key, shape=(self.batch_size,), minval=0, maxval=state.size)

        # Gather the data at the sampled indices. This is efficient on GPU.
        batch_data = jax.tree_util.tree_map(
            lambda buffer_leaf: buffer_leaf[indices],
            state.data
        )
        
        # Convert the Transition PyTree to a dictionary to match the original API.
        return {
            "observations": batch_data.observation,
            "actions": batch_data.action,
            "rewards": batch_data.reward,
            "next_observations": batch_data.next_observation,
            "terminals": batch_data.terminal,
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def flush(self, state: BufferState) -> BufferState:
        """
        Resets the buffer to an empty state by resetting the pointer and size.

        Args:
            state: The current BufferState.

        Returns:
            A new, empty BufferState.
        """
        return state.replace(
            pointer=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32)
        )

if __name__ == "__main__":
    # --- Example Setup ---
    CAPACITY = 10000
    OBS_DIM = 4
    ACTION_DIM = 1
    BATCH_SIZE = 256

    # 1. Instantiate the buffer (stateless object)
    replay_buffer = JaxReplayBuffer(capacity=CAPACITY)

    # 2. Create a prototype transition to define shapes and dtypes
    prototype = Transition(
        observation=jnp.zeros(OBS_DIM, dtype=jnp.float32),
        action=jnp.zeros(ACTION_DIM, dtype=jnp.float32),
        reward=jnp.zeros((), dtype=jnp.float32),
        next_observation=jnp.zeros(OBS_DIM, dtype=jnp.float32),
        terminal=jnp.zeros((), dtype=jnp.bool_)
    )

    # 3. Initialize the buffer state
    buffer_state = replay_buffer.init(prototype)

    # 4. Add some data (e.g., in a loop)
    # In a real application, this would come from your environment.
    for i in range(500):
        dummy_transition = Transition(
            observation=jnp.full(OBS_DIM, i, dtype=jnp.float32),
            action=jnp.full(ACTION_DIM, i, dtype=jnp.float32),
            reward=jnp.array(i, dtype=jnp.float32),
            next_observation=jnp.full(OBS_DIM, i + 1, dtype=jnp.float32),
            terminal=jnp.array(i == 499)
        )
        # The add function returns a *new* state
        buffer_state = replay_buffer.add(buffer_state, dummy_transition)

    print(f"Buffer size after adding data: {buffer_state.size}")

    # 5. Sample a batch of data
    # Create a JAX random key
    key = jax.random.PRNGKey(0)
    key, sample_key = jax.random.split(key)

    # Sample the buffer
    batch = replay_buffer.sample(buffer_state, BATCH_SIZE, sample_key)

    print(f"\nSampled batch shapes:")
    for name, arr in batch.items():
        print(f"- {name}: {arr.shape}")

    # 6. Flush the buffer
    buffer_state = replay_buffer.flush(buffer_state)
    print(f"\nBuffer size after flushing: {buffer_state.size}")