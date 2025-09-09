from flax import nnx
from typing import Sequence, Callable
import jax.numpy as jnp

class DeterministicCritic(nnx.Module):
    """
    A deterministic critic network implemented using Flax NNX.

    This network takes a state-action representation as input and outputs a
    single Q-value. It's composed of a series of dense layers with optional
    layer normalization and dropout.
    """
    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        *,  # Make rngs a keyword-only argument
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        """
        Initializes the DeterministicCritic module.

        Args:
            in_features: The number of input features (e.g., state_dim + action_dim).
            features: A sequence of integers defining the number of units in each hidden layer.
            rngs: The random number generators for initializing weights and for dropout.
            activation_fn: The activation function to use for hidden layers.
            use_layer_norm: If True, adds a LayerNorm layer after each hidden dense layer.
            dropout_rate: The dropout rate to apply after activation in hidden layers.
                          If 0.0, no dropout is applied.
        """
        # --- Store static configuration ---
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn

        # --- Define stateful layers ---
        self.hidden_layers = []
        if self.use_layer_norm:
            self.norm_layers = []
        
        # Create hidden layers dynamically
        current_features = in_features
        for feat in features:
            self.hidden_layers.append(nnx.Linear(current_features, feat, rngs=rngs))
            if self.use_layer_norm:
                self.norm_layers.append(nnx.LayerNorm(feat, rngs=rngs))
            current_features = feat

        # Dropout and output layers
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.output_layer = nnx.Linear(current_features, 1, rngs=rngs)

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Performs the forward pass of the critic network.

        Args:
            x: The input tensor, typically a concatenation of observations and actions.
            training: If True, dropout layers are active. Otherwise, they are in
                   inference mode (deterministic).

        Returns:
            The scalar Q-value for the input, with shape (batch_size,).
        """
        # Concatenate observations and actions to form the input to the network
        x = jnp.concatenate([observations, actions], axis=-1)
        # Use the layers defined in __init__
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)
            x = self.dropout(x, deterministic=not training)
        
        x = self.output_layer(x)
        return x
    

def StochasticCritic():
    def __init__(self):
        pass

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        pass