from mujoco_playground import registry
from mujoco_playground import wrapper
from typing import Any, Callable, Optional
import jax.numpy as jnp
from mujoco_playground._src import mjx_env
from flax import struct

@struct.dataclass
class WrapperState:
    """A dataclass to hold the state for the jittable wrapper.

    This makes state explicit, containing the original environment state
    and the wrapper's step counter.
    """
    env_state: mjx_env.State
    step_count: jnp.ndarray



class TerminationWrapper(wrapper.Wrapper):
    """
    Non-invasive wrapper that extends mujoco_playground's base Wrapper.
    
    Key principle: Never modify the original playground state.
    Instead, we return a tuple of (original_state, wrapper_info) or use
    a separate tracking mechanism.
    """
    
    def __init__(
        self,
        env: Any,
        max_episode_steps: int = 1000,
    ):
        super().__init__(env)
        
        self.max_episode_steps = max_episode_steps
        
        # External tracking (not part of state)
        self._current_step_count = 0
        self._episode_active = False
    
    def reset(self, key: jnp.ndarray) -> WrapperState:
        """Resets the environment and the wrapper's state."""
        # Reset the base environment to get the initial mjx_env.State
        initial_env_state = super().reset(key)

        # Update the info dictionary for observation purposes
        new_info = initial_env_state.info | {
            'truncation': False,
            'termination': False,
        }

        # Create the final environment state with the updated done flag and info
        initial_env_state = initial_env_state.replace(info=new_info)

        # Return the initial WrapperState, starting the step count at 0
        return WrapperState(
            env_state=initial_env_state,
            step_count=jnp.zeros((), dtype=jnp.int32),
        )

    def step(self, state: WrapperState, action: jnp.ndarray) -> WrapperState:
        """
        Performs a step in the environment. This function is now pure
        and can be safely jitted.
        """
        # Step the underlying environment using its state
        next_env_state = super().step(state.env_state, action)
        
        # Increment the step count from the input state
        new_step_count = state.step_count + 1

        # Determine truncation based on the new step count
        truncated = new_step_count >= self.max_episode_steps

        # The episode is done if the base environment terminates OR if it's truncated.
        # next_env_state.done is the termination signal from the base env.
        done = jnp.logical_or(next_env_state.done, truncated)
        
        # Update the info dictionary for observation purposes
        new_info = next_env_state.info | {
            'truncation': truncated,
            'termination': next_env_state.done,
        }

        # Create the final environment state with the updated done flag and info
        final_env_state = next_env_state.replace(done=done, info=new_info)

        # Return the new WrapperState containing the new env state and step count
        return WrapperState(
            env_state=final_env_state,
            step_count=new_step_count
        )




def load_playground_env(env_name: str):
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    wrapped_env = TerminationWrapper(env)
    return wrapped_env, env_cfg