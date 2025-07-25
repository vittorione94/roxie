from myojit.agents import agent
import jax
import jax.numpy as jnp

class DDPG(agent.Agent):
    def __init__(self, actor, critic, replay, exploration=None, actor_updater=None,
        critic_updater=None):
        self.actor = actor
        self.critic = critic
        self.replay = replay
        self.action_low = -1
        self.action_high = 1
        

    @jax.jit
    def step(self, state: jnp.ndarray, evaluate: bool = False) -> jnp.ndarray:
        """Selects an action, adding noise for exploration if not in evaluation mode."""
        
        # Get the deterministic action from the actor network
        action = self.actor(state)
        
        # Use jax.lax.cond for conditional logic inside a JIT-compiled function
        # action = jax.lax.cond(
        #     evaluate,
        #     # If evaluate is True, return the action as is
        #     lambda: action,
        #     # If evaluate is False, add Gaussian noise
        #     lambda: action + jax.random.normal(key, action.shape) * self.cfg.exploration_noise
        # )
        
        # Clip the final action to be within the environment's valid bounds
        return jnp.clip(action, self.action_low, self.action_high)
    
    def update(observations, rewards, resets, terminations, steps):
        


    # def save():
    #     _, state = nnx.split(model)
    #     checkpointer = ocp.StandardCheckpointer()
    #     checkpointer.save(ckpt_dir / 'state', state)

    # def load():
    #     abstract_model = nnx.eval_shape(lambda: TwoLayerMLP(4, rngs=nnx.Rngs(0)))
    #     graphdef, abstract_state = nnx.split(abstract_model)

    #     state_restored = checkpointer.restore(ckpt_dir / 'state', abstract_state)
    #     jax.tree.map(np.testing.assert_array_equal, state, state_restored)

    #     model = nnx.merge(graphdef, state_restored)