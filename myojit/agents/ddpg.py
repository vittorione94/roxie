from myojit.agents import agent
import jax
import jax.numpy as jnp
import functools
from myojit.replays.buffer import Transition, JaxReplayBuffer
from flax import nnx
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Union
import optax
import copy
from typing import Any


class TrainState(nnx.Module):
  def __init__(
      self,
      *,
      actor: nnx.Module,
      critic: nnx.Module,
      target_actor: nnx.Module,
      target_critic: nnx.Module,
      actor_optimizer: nnx.Optimizer,
      critic_optimizer: nnx.Optimizer,
      buffer_state: Any,
  ):
    """Initializes the training state.

    The state components are defined as attributes of the module.
    `nnx.Module` will automatically know how to handle them for
    JAX transformations.
    """
    self.actor = actor
    self.critic = critic
    self.target_actor = target_actor
    self.target_critic = target_critic

    self.actor_optimizer = actor_optimizer
    self.critic_optimizer = critic_optimizer

    self.buffer_state = buffer_state


def _critic_loss_fn(critic_model, target_actor_model, target_critic_model, samples, gamma):
    """Calculates the MSE loss for the critic."""
    next_actions = target_actor_model(samples['next_observations'])
    next_q = target_critic_model(samples['next_observations'], next_actions)
    
    target_q = samples['rewards'] + gamma * (1.0 - samples['terminals']) * jnp.squeeze(next_q)
    
    current_q = critic_model(samples['observations'], samples['actions'])
    
    critic_loss = jnp.mean((jnp.squeeze(current_q) - target_q)**2)
    return critic_loss

def _actor_loss_fn(actor_model, critic_model, samples):
    """Calculates the loss for the actor (aims to maximize Q-value)."""
    actions = actor_model(samples['observations'])
    q_values = critic_model(samples['observations'], actions)
    actor_loss = -jnp.mean(q_values)
    return actor_loss


# This is the core computational kernel that will be JIT-compiled.
@functools.partial(nnx.jit, static_argnames=('gamma', 'tau', 'replay_sample_fn'))
def _grad_step(state: TrainState, key: jax.random.PRNGKey, gamma: float, tau: float, replay_sample_fn):
    """Performs one full gradient update step and returns the new state."""
    # 1. Sample from the replay buffer
    samples = replay_sample_fn(state.buffer_state, key)

    # 2. Calculate critic gradients and update the critic
    critic_grads = nnx.grad(_critic_loss_fn)(
        state.critic, state.target_actor, state.target_critic, samples, gamma
    )
    state.critic_optimizer.update(critic_grads)
    
    # 3. Calculate actor gradients and update the actor
    # Use the *original* critic for the actor loss calculation, not the updated one
    actor_grads = nnx.grad(_actor_loss_fn)(
        state.actor, state.critic, samples
    )
    state.actor_optimizer.update(actor_grads)
    
    # 4. Update target networks using soft updates
    new_actor_tensors = nnx.state(state.actor, nnx.Param)  # Use updated_actor
    old_actor_tensors = nnx.state(state.target_actor, nnx.Param)

    new_target_actor_tensors = optax.incremental_update(
          new_tensors=new_actor_tensors,
          old_tensors=old_actor_tensors,
          step_size=tau)
    
    new_critic_tensors = nnx.state(state.critic, nnx.Param)
    old_critic_tensors = nnx.state(state.target_critic, nnx.Param)
    new_target_critic_tensors = optax.incremental_update(
          new_tensors=new_critic_tensors,
          old_tensors=old_critic_tensors,
          step_size=tau)
    

    nnx.update(state.target_actor, new_target_actor_tensors)
    nnx.update(state.target_critic, new_target_critic_tensors)

    # 5. Return the new, updated state object
    return TrainState(
            actor=state.actor,
            critic=state.critic,
            actor_optimizer=state.actor_optimizer,
            target_actor=state.target_actor,
            target_critic=state.target_critic,
            critic_optimizer=state.critic_optimizer,
            buffer_state=state.buffer_state
        ), samples["indices"]

@functools.partial(jax.jit, static_argnames=('actor_model', 'evaluate',))
def _step_fn(
    actor_model,
    observation: jnp.ndarray,
    key: jax.random.PRNGKey,
    exploration_noise: float,
    action_low: float,
    action_high: float,
    evaluate: bool = False,
):
    """
    A pure function to select an action.
    Its output depends only on its inputs.
    """
    # Get the deterministic action from the actor network.
    action = actor_model(observation)

    # Generate noise unconditionally to keep the computation graph static.
    noise = jax.random.normal(key, action.shape)
    
    # Use jax.lax.cond to select the noise scale based on the static 'evaluate' flag.
    noise_scale = jax.lax.cond(
        evaluate,
        lambda: 0.0,                   # If evaluating, scale is 0.
        lambda: exploration_noise      # If training, use the defined scale.
    )
    
    _n = noise * noise_scale
    # Apply the scaled noise.
    noisy_action = action + _n
            
    # Clip the final action to be within the environment's valid bounds.
    return jnp.clip(noisy_action, action_low, action_high), _n

# @functools.partial(nnx.jit, static_argnames=('gamma', 'tau', 'replay_sample_fn', 'num_steps'))
# def _multiple_grad_steps_nnx(
#     state: TrainState, 
#     key: jax.random.PRNGKey, 
#     gamma: float, 
#     tau: float, 
#     replay_sample_fn,
#     num_steps: int
# ):
#     """Performs multiple gradient update steps using nnx.fori_loop."""
    
#     def single_step(i, carry):
#         current_state, current_key = carry
#         current_key, subkey = jax.random.split(current_key)
#         updated_state, _ = _grad_step(current_state, subkey, gamma, tau, replay_sample_fn)
#         return updated_state, current_key
    
#     final_state, _ = nnx.fori_loop(0, num_steps, single_step, (state, key))
#     return final_state

class DDPG(agent.Agent):
    def __init__(self, 
                 actor: nnx.Module, 
                 critic: nnx.Module, 
                 replay: JaxReplayBuffer, 
                 buffer_state: 'BufferState', 
                 *,
                 actor_learning_rate: float = 3e-4,
                 critic_learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 exploration_noise: float = 0.1,
                 steps_before_learning: int = 10000,
                 steps_between_updates: int = 500,
                 learning_steps: int = 100,
                 memory_warmup: int = 10000,
                 ):

        # create targets
        target_actor = copy.deepcopy(actor)
        target_critic = copy.deepcopy(critic)
        
        actor_optimizer = nnx.Optimizer(
            actor, optax.adam(actor_learning_rate), wrt=nnx.Param
        )

        critic_optimizer = nnx.Optimizer(
            critic, optax.adam(critic_learning_rate), wrt=nnx.Param
        )

        self.state = TrainState(
            actor=actor,
            critic=critic,
            target_actor=target_actor,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            buffer_state=buffer_state, # Assuming buffer has an init method
        )

        # --- Store hyperparameters ---
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.action_low = -1
        self.action_high = 1
        self.replay = replay
        self.steps_before_learning = steps_before_learning
        self.steps_between_updates = steps_between_updates
        self.learning_steps = learning_steps
        self.memory_warmup = memory_warmup


    def step(self, observation: jnp.ndarray, evaluate: bool = False, key: jax.random.PRNGKey = None) -> jnp.ndarray:
        """
        Selects an action by calling the pure, JIT-compiled step function.
        """
        # Call the standalone function, passing in the required parts from the agent's state.
        action, noise = _step_fn(
            self.state.actor,
            observation,
            key,
            self.exploration_noise,
            self.action_low,
            self.action_high,
            evaluate,
        )
        
        return action
    
    def update(self, prev_states, states, steps, agent_rng):
        experiences = Transition(
            observation=prev_states.obs,
            action=states.data.ctrl,
            reward=states.reward,
            next_observation=states.obs,
            terminal=states.done,
        )
        # store in memory
        self.state.buffer_state = self.replay.add_batch(self.state.buffer_state, experiences)

        # Conditionally call the JIT-compiled gradient step
        if steps > self.steps_before_learning and steps % self.steps_between_updates == 0:
            for i in range(self.learning_steps):
                agent_rng, key = jax.random.split(agent_rng, 2)

                self.state, _ = _grad_step(
                    self.state, 
                    key, 
                    self.gamma, 
                    self.tau,
                    self.replay.sample # Pass the sample method itself
                )

                # Perform multiple gradient steps in a single JIT-compiled call
            # self.state = _multiple_grad_steps_nnx(
            #     self.state, 
            #     key, 
            #     self.gamma, 
            #     self.tau,
            #     self.replay.sample,
            #     self.learning_steps
            # )


        return self.state.buffer_state

    def save(self, path: Union[str, Path]):
        """
        Saves the complete state of the agent to a directory.

        This includes the state of the actor network, the critic network,
        and the replay buffer.

        Args:
            path: The directory path where the checkpoint will be saved.
        """
        # Ensure the path is a Path object
        path = Path(path).resolve()
        # path.mkdir(parents=True, exist_ok=True)
        
        # Split models into static graph definition and dynamic state
        _, actor_state = nnx.split(self.state.actor)
        _, critic_state = nnx.split(self.state.critic)

        # Create a Pytree containing all the data to save
        save_data = {
            'actor': actor_state,
            'critic': critic_state,
            'buffer': self.state.buffer_state
        }
        
        # Use Orbax to save the Pytree
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, save_data)
        print(f"Agent state saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], actor: nnx.Module, critic: nnx.Module, replay: JaxReplayBuffer):
        """
        Loads the agent's state from a directory.

        This method restores the state of the actor, critic, and replay buffer.
        It requires placeholder instances of the models and replay buffer to
        populate them with the loaded state.

        Args:
            path: The directory path of the saved checkpoint.
            actor: An instance of the actor model with the correct architecture.
            critic: An instance of the critic model with the correct architecture.
            replay: An instance of the replay buffer.

        Returns:
            A new DDPG agent instance with the loaded state.
        """
        # --- FIX: Also resolve the path for loading for consistency ---
        path = Path(path).resolve()
        
        # Use Orbax to load the Pytree
        checkpointer = ocp.PyTreeCheckpointer()
        
        # Restore the saved data structure. restore() can infer the structure
        # from the provided path.
        loaded_data = checkpointer.restore(path)
        
        # Merge the loaded state back into the model objects
        # To merge state into a model, we first need its structure (GraphDef).
        actor_graphdef, _ = nnx.split(actor)
        critic_graphdef, _ = nnx.split(critic)

        # Then we can merge the structure with the loaded state to create new models.
        actor = nnx.merge(actor_graphdef, loaded_data['actor'])
        critic = nnx.merge(critic_graphdef, loaded_data['critic'])
        buffer_state = loaded_data['buffer']
        
        print(f"Agent state loaded from {path}")
        
        # Create and return a new agent instance with the restored components
        return cls(actor=actor, critic=critic, replay=replay, buffer_state=buffer_state)
