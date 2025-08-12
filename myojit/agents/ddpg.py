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


def _critic_loss_fn(critic_model, target_actor_model, target_critic_model, samples, \
                    gamma, noise_key, target_policy_noise, target_noise_clip,  action_low, action_high):
    """Calculates the MSE loss for the critic."""
    next_actions = target_actor_model(samples['next_observations'])

    noise = jax.random.normal(noise_key, next_actions.shape) * target_policy_noise
    noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    next_actions = next_actions + noise
    # Clip actions to valid range (assuming -1 to 1)
    next_actions = jnp.clip(next_actions, action_low, action_high)

    next_q = target_critic_model(samples['next_observations'], next_actions)
    
    term = samples['terminals'].astype(jnp.float32)
    reward = jnp.squeeze(samples['rewards'])
    next_q = jnp.squeeze(next_q)
    target_q = reward + gamma * (1.0 - term) * next_q
    target_q = jax.lax.stop_gradient(target_q)

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
def _grad_step(state: TrainState, key: jax.random.PRNGKey, gamma: float, tau: float, replay_sample_fn, \
               exploration_noise: float, target_policy_noise: float, target_noise_clip: float, action_low: float, action_high: float):
    """Performs one full gradient update step and returns the new state."""
    # 1. Sample from the replay buffer
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)

    # 2. Calculate critic gradients and update the critic
    critic_loss, critic_grads = nnx.value_and_grad(_critic_loss_fn)(
        state.critic, state.target_actor, state.target_critic, samples, gamma, noise_key, \
            target_policy_noise, target_noise_clip, action_low, action_high
    )
    state.critic_optimizer.update(critic_grads)
    
    # 3. Calculate actor gradients and update the actor
    # Use the *original* critic for the actor loss calculation, not the updated one
    actor_loss, actor_grads = nnx.value_and_grad(_actor_loss_fn)(
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
        ), actor_loss, critic_loss


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
                #  exploration_noise: float = 0.1,
                 init_noise: float = 0.1,
                 min_noise: float = 0.01,
                 noise_decay_transitions: int = 100000,
                 steps_before_learning: int = 100,
                 steps_between_updates: int = 10,
                 learning_steps: int = 5,
                 memory_warmup: int = 100,
                 target_noise_clip: float = 0.1,
                 target_policy_noise: float = 0.1,
                 max_grad_norm: float = 1.0,
                 action_low: float = -1.0,
                 action_high: float = 1.0
                 ):

        # create targets
        target_actor = copy.deepcopy(actor)
        target_critic = copy.deepcopy(critic)
        
        # Add gradient clipping to optimizers
        actor_optimizer = nnx.Optimizer(
            actor, 
            optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(actor_learning_rate)
            ), 
            wrt=nnx.Param
        )

        critic_optimizer = nnx.Optimizer(
            critic, 
            optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(critic_learning_rate)
            ), 
            wrt=nnx.Param
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
        self.exploration_noise = init_noise
        self.init_noise = init_noise
        self.noise_decay_transitions = noise_decay_transitions
        self.min_noise = min_noise
        self.action_low = action_low
        self.action_high = action_high
        self.replay = replay
        self.steps_before_learning = steps_before_learning
        self.steps_between_updates = steps_between_updates
        self.learning_steps = learning_steps
        self.memory_warmup = memory_warmup
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        print("DDPG agent initialized.")
        print("Params: \n" \
        f"   gamma {self.gamma} \n" \
        f"   tau {self.tau}\n" \
        f"   exploration_noise {self.exploration_noise}\n" \
        f"   steps_before_learning {self.steps_before_learning}\n" \
        f"   steps_between_updates {self.steps_between_updates}\n" \
        f"   learning_steps {self.learning_steps}\n" \
        f"   memory_warmup {self.memory_warmup}\n" \
        f"   target_noise_clip {self.target_noise_clip}\n" \
        f"   action_low {self.action_low}\n" \
        f"   action_high {self.action_high}\n" \
        f"   max_grad_norm {max_grad_norm}\n" \
        f"   actor_learning_rate {actor_learning_rate}\n" \
        f"   critic_learning_rate {critic_learning_rate}\n" \
        f"   target_policy_noise {target_policy_noise}\n" )


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
    
    def _decay_noise(self, steps):
        f = min(1.0, steps / self.noise_decay_transitions)
        self.exploration_noise = float(self.init_noise - (self.init_noise - self.min_noise) * f)


    def update(self, prev_states, states, steps, agent_rng, actions):
        experiences = Transition(
            observation=prev_states.obs,
            action=actions,
            reward=states.reward,
            next_observation=states.obs,
            terminal=states.done,
        )
        # store in memory
        self.state.buffer_state = self.replay.add_batch(self.state.buffer_state, experiences)

        gradient_steps = 0
        actor_loss, critic_loss = 0, 0

        self._decay_noise(steps)

        # Conditionally call the JIT-compiled gradient step
        if steps >= self.steps_before_learning and \
           (steps - self.steps_before_learning) % self.steps_between_updates == 0:
            for _ in range(self.learning_steps):
                agent_rng, key = jax.random.split(agent_rng, 2)

                self.state, actor_loss, critic_loss = _grad_step(
                    self.state, 
                    key, 
                    self.gamma, 
                    self.tau,
                    self.replay.sample, # Pass the sample method itself,
                    self.exploration_noise,
                    self.target_policy_noise,  # Use target_policy_noise
                    self.target_noise_clip,
                    self.action_low,
                    self.action_high
                )
            gradient_steps += self.learning_steps
                # Perform multiple gradient steps in a single JIT-compiled call
            # self.state = _multiple_grad_steps_nnx(
            #     self.state, 
            #     key, 
            #     self.gamma, 
            #     self.tau,
            #     self.replay.sample,
            #     self.learning_steps
            # )


        return gradient_steps, actor_loss, critic_loss


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
        
            # Extract parameters and move to CPU
        _, actor_state = nnx.split(self.state.actor)
        _, critic_state = nnx.split(self.state.critic)
        _, target_actor_state = nnx.split(self.state.target_actor)
        _, target_critic_state = nnx.split(self.state.target_critic)
        
        save_data = {
            'actor': jax.device_get(actor_state),
            'critic': jax.device_get(critic_state),
            'target_actor': jax.device_get(target_actor_state),
            'target_critic': jax.device_get(target_critic_state),
            'buffer': jax.device_get(self.state.buffer_state),
            'hyperparams': {
                'gamma': self.gamma,
                'tau': self.tau,
                'exploration_noise': self.exploration_noise,
                'target_noise_clip': self.target_noise_clip,
                'action_low': self.action_low,
                'action_high': self.action_high
            }
        }
        
        checkpointer = ocp.StandardCheckpointer()  # More device-agnostic
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
