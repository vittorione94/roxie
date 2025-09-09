import jax
import jax.numpy as jnp
import flax.struct as struct
from typing import Dict
import functools
from typing import Dict, Any, Optional

# A PyTree representing a single data transition (s, a, r, s', d).
# This structure is JAX-native and can be passed into JIT-compiled functions.
@struct.dataclass
class Transition:
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_observation: jnp.ndarray
    terminal: jnp.ndarray
    # log_probs: Optional[jnp.ndarray] = None  # Optional, used in some algorithms

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

    def __init__(self, capacity: int, batch_size: int, num_envs: int = 1):
        """
        Initializes the replay buffer with a fixed capacity.
        Note: This class is stateless. It only holds static configuration.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.num_envs = num_envs

        print("Replay buffer initialized.")
        print("Params: \n" \
        f"   capacity {self.capacity} \n" \
        f"   batch_size {self.batch_size}\n" \
        f"   num_envs {self.num_envs}\n")


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
    def add_batch(self, state: BufferState, 
                  experiences: Transition) -> BufferState:
        """
        Adds a batch of new experiences (a PyTree) to the buffer.

        Args:
            state: The current BufferState.
            experiences: A PyTree of experiences to add. Each leaf must have a
                         leading batch dimension and match the structure of the
                         prototype used in `init`.

        Returns:
            A new BufferState with the batch of experiences added.
        """

        # The logic is generic and works on any PyTree structure.
        batch_size = jax.tree_util.tree_leaves(experiences)[0].shape[0]
        indices = (jnp.arange(batch_size) + state.pointer) % self.capacity

        updated_data = jax.tree_util.tree_map(
            lambda buffer_leaf, experience_batch: buffer_leaf.at[indices].set(experience_batch),
            state.data,
            experiences
        )

        new_pointer = (state.pointer + batch_size) % self.capacity
        new_size = jnp.minimum(state.size + batch_size, self.capacity)

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
            "indices": indices
        }
    
    # On-policy: fetch last T steps in time-major order [T, B, ...] + bootstrap obs [B, ...]
    def get_recent_window(self, state: BufferState, T: int) -> Dict[str, jnp.ndarray]:
        """
        Returns:
            dict with keys observations/actions/rewards/next_observations/terminals/log_probs
            each shaped [T, B, ...], and 'bootstrap_observation' shaped [B, ...].
        Assumes each add_batch wrote exactly num_envs rows (one per env).
        """

        B = self.num_envs
        assert (int(state.size) % B) == 0, "Buffer size must be a multiple of num_envs"
        steps_written = int(state.size) // B

        # effective rollout length
        T_eff = min(int(T), steps_written)

        # pointer is in slots; convert to 'step units'
        step_pointer = int(state.pointer) // B  # next step index to write
        # the window occupies steps [step_pointer - T_eff, ..., step_pointer - 1] modulo max_steps
        max_steps = self.capacity // B
        start_step = (step_pointer - T_eff) % max_steps
        start_slot = (start_step * B) % self.capacity

        # flat slot indices for the window (wrap-around safe)
        idx = (jnp.arange(T_eff * B) + start_slot) % self.capacity

        def gather_reshape(x):
            xb = x[idx]                                # [T_eff * B, ...]
            return xb.reshape(T_eff, B, *xb.shape[1:]) # [T_eff, B, ...]

        window = jax.tree_util.tree_map(gather_reshape, state.data)

        # Bootstrap comes from the last time slice's next_observation
        bootstrap_observation = window.next_observation[-1]  # [B, ...]

        return {
            "observations":       window.observation,       # [T, B, ...]
            "actions":            window.action,            # [T, B, ...]
            "rewards":            window.reward,            # [T, B]
            "next_observations":  window.next_observation,  # [T, B, ...]
            "terminals":          window.terminal,          # [T, B]
            "log_probs":          window.log_probs,         # [T, B, ...] or [T,B]
            "bootstrap_observation": bootstrap_observation, # [B, ...]
            "T_eff":              jnp.array(T_eff, dtype=jnp.int32),
            "B":                  jnp.array(B, dtype=jnp.int32),
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