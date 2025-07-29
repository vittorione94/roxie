from myojit.agents import agent
import jax
import jax.numpy as jnp
import functools
from myojit.replays.buffer import Transition, JaxReplayBuffer
import flax.experimental.nnx as nnx
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Union
import optax
import copy

class DDPG(agent.Agent):
    def __init__(self, 
                 actor: nnx.Module, 
                 critic: nnx.Module, 
                 replay: JaxReplayBuffer, 
                 buffer_state: 'BufferState', 
                 *,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 exploration_noise: float = 0.1):
        self.actor = actor
        self.critic = critic

        # create targets
        self.target_actor = copy.deepcopy(actor)
        self.target_critic = copy.deepcopy(critic)

        self.replay = replay
        self.buffer_state = buffer_state
        
        # --- Store hyperparameters ---
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.action_low = -1
        self.action_high = 1

        self.actor_optimizer = nnx.Optimizer(
            self.actor, optax.adam(learning_rate), wrt=nnx.Param
        )

        self.critic_optimizer = nnx.Optimizer(
            self.critic, optax.adam(learning_rate), wrt=nnx.Param
        )

    @functools.partial(jax.jit, static_argnums=(0, 2, ))
    def step(self, state: jnp.ndarray, evaluate: bool = False, key: jax.random.PRNGKey = None) -> jnp.ndarray:
        """Selects an action, adding noise for exploration if not in evaluation mode."""
        
        # Get the deterministic action from the actor network.
        # The 'dropout' RNG stream will be used internally by the actor if training=True.
        # This part is fine as nnx handles the context correctly.
        action = self.actor(state)

        # 1. Generate noise UNCONDITIONALLY.
        # This ensures an RNG key from the 'agent' stream is always used,
        # making the computation graph static and JIT-compatible.
        noise = jax.random.normal(key, action.shape)
        
        # 2. Use jax.lax.cond to select the SCALE of the noise.
        # This operates on simple float values and is safe to JIT.
        noise_scale = jax.lax.cond(
            evaluate,
            lambda: 0.0,                   # If evaluating, scale noise by 0.
            lambda: self.exploration_noise # If training, use the defined noise scale.
        )
        
        # 3. Apply the scaled noise to the action.
        # If evaluating, this is equivalent to `action + 0`.
        noisy_action = action + noise * noise_scale
                
        # Clip the final action to be within the environment's valid bounds.
        return jnp.clip(noisy_action, self.action_low, self.action_high)
    
    def update(self, prev_states, states, steps, key):
        
        experiences = Transition(
            observation=prev_states.obs,
            action=states.data.ctrl,
            reward=states.reward,
            next_observation=states.obs,
            terminal=states.done,
        )
        # store in memory
        self.buffer_state = self.replay.add_batch(self.buffer_state, experiences)

        # If steps are enough update:
        samples = self.replay.sample(self.buffer_state, key)

        #1. update actor 
        # This function takes the actor module whose parameters we want to differentiate.
        def actor_loss_fn(actor_to_grad, critic):
            # Use the actor passed to the function to get actions
            actions = actor_to_grad(samples['observations'])
            
            # The critic's parameters are treated as constant here, which is correct for an actor update.
            q_values = critic(samples['observations'], actions)
            
            # The loss is the negative mean of the Q-values, which we want to maximize.
            loss = -jnp.mean(q_values)
            return loss

        # 2. Use nnx.grad to compute gradients for the actor's parameters.
        # nnx.grad is designed to work with NNX modules and correctly handles their state.
        actor_grads = nnx.grad(actor_loss_fn)(self.actor, self.critic)
        self.actor_optimizer.update(actor_grads)


        #2. update critic
        def critic_loss_fn(critic_to_grad, target_actor, target_critic):
            # Get next actions from the TARGET actor, not the main one.
            next_actions = target_actor(samples['next_observations'])
            
            # Get Q-values for the next states from the TARGET critic.
            next_q_values = target_critic(samples['next_observations'], next_actions)
            
            # Compute the Bellman target.
            target_q_values = samples['rewards'] + self.gamma * (1.0 - samples['terminals']) * jnp.squeeze(next_q_values)
            
            # Get current Q-values from the critic we are differentiating.
            q_values = critic_to_grad(samples['observations'], samples['actions'])
            
            # Compute the MSE loss.
            loss = jnp.mean((jnp.squeeze(q_values) - target_q_values)**2)
            return loss

        critic_grads = nnx.grad(critic_loss_fn)(self.critic, self.target_actor, self.target_critic)
        self.critic_optimizer.update(critic_grads)

        # 3. Soft-update target networks
        self.soft_update(self.tau, self.actor, self.target_actor)
        self.soft_update(self.tau, self.critic, self.target_critic)
                         
        return self.buffer_state

    def soft_update(self, tau: float, online_model: nnx.Module, target_model: nnx.Module):
        """
        Performs a soft update of the target model's parameters from the online model's parameters.

        This function modifies the `target_model` in place.

        Args:
            tau: The interpolation parameter (typically a small value like 0.005).
            online_model: The model being trained directly (source of new parameters).
            target_model: The model to be updated slowly (destination).
        """
        # 1. Get the parameters from both the online and target models.
        #    nnx.state() returns a pytree of the model's state, which we filter for just nnx.Param.
        online_params = nnx.state(online_model, nnx.Param)
        target_params = nnx.state(target_model, nnx.Param)

        # 2. Use jax.tree_util.tree_map to apply the soft update formula to each parameter.
        #    The lambda function is applied element-wise to each parameter tensor in the pytrees.
        new_target_params = jax.tree_util.tree_map(
            lambda online, target: tau * online + (1 - tau) * target,
            online_params,
            target_params
        )

        # 3. Update the target model with the new parameters.
        #    nnx.update() applies the changes from the pytree back to the model object.
        nnx.update(target_model, new_target_params)

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
        _, actor_state = nnx.split(self.actor)
        _, critic_state = nnx.split(self.critic)

        # Create a Pytree containing all the data to save
        save_data = {
            'actor': actor_state,
            'critic': critic_state,
            'buffer': self.buffer_state
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
