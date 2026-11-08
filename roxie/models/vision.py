"""Pixel encoder that lets every actor and critic perceive images.

Observations stay flat everywhere outside the networks — the replay buffer, the
running observation statistics, the rollout and the losses all see one vector
per world. A pixel observation rides at the FRONT of that vector, flattened from
`(H, W, C)` in row-major order, with any proprioceptive state after it. The env
advertises the image's shape as `pixel_shape`, which `publish_env_shapes` writes
to `env.pixel_shape` for the configs to interpolate.

A network built with a `vision` block reshapes that leading block back into an
image, runs it through `PixelEncoder`, and hands the MLP the encoder's latent
followed by the untouched remainder. Each network owns its encoder: the actor,
the critic and each twin head learn their own from their own loss. That costs a
conv stack per network rather than one shared one, and buys an encoder that
works unchanged under every agent's target copies, optimizers and checkpoints.
"""

import math
from typing import Callable, Mapping, Optional, Sequence

import jax.numpy as jnp
from flax import nnx

from roxie.utils.precision import as_float, linear_kwargs


class PixelEncoder(nnx.Module):
    """A CNN embedding the image that leads a flat observation.

    Attributes:
        pixel_shape: `(H, W, C)` of the image block.
        latent_dim: width of the embedding that replaces the image block.
    """

    def __init__(
        self,
        pixel_shape: Sequence[int],
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        latent_dim: int,
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
    ):
        if pixel_shape is None:
            raise ValueError(
                "a `vision` block was configured, but the env publishes no "
                "`pixel_shape`: it produces no pixel observations to encode"
            )
        if not len(channels) == len(kernel_sizes) == len(strides):
            raise ValueError(
                f"channels {list(channels)}, kernel_sizes {list(kernel_sizes)} "
                f"and strides {list(strides)} must name one entry per conv layer"
            )
        self.pixel_shape = tuple(int(d) for d in pixel_shape)
        if len(self.pixel_shape) != 3:
            raise ValueError(
                f"pixel_shape must be (H, W, C), got {self.pixel_shape}"
            )
        self.latent_dim = int(latent_dim)
        self.activation_fn = activation_fn

        height, width, current = self.pixel_shape
        conv_layers = []
        for out_channels, kernel, stride in zip(channels, kernel_sizes, strides):
            conv_layers.append(
                nnx.Conv(
                    current, int(out_channels),
                    kernel_size=(int(kernel), int(kernel)),
                    strides=(int(stride), int(stride)),
                    padding="VALID",
                    rngs=rngs,
                    **linear_kwargs(),
                )
            )
            height = (height - int(kernel)) // int(stride) + 1
            width = (width - int(kernel)) // int(stride) + 1
            current = int(out_channels)
            if height < 1 or width < 1:
                raise ValueError(
                    f"the conv stack shrinks a {self.pixel_shape} image to "
                    f"nothing; use smaller kernels or strides"
                )
        self.conv_layers = nnx.List(conv_layers)

        self.projection = nnx.Linear(
            height * width * current, self.latent_dim, rngs=rngs,
            **linear_kwargs(),
        )
        self.norm = nnx.LayerNorm(self.latent_dim, rngs=rngs)

    @property
    def pixel_size(self) -> int:
        """The number of flat observation entries the image occupies."""
        return math.prod(self.pixel_shape)

    def out_features(self, in_features: int) -> int:
        """The width a flat input of `in_features` has once encoded."""
        if in_features < self.pixel_size:
            raise ValueError(
                f"a {in_features}-wide input cannot hold a {self.pixel_shape} "
                f"image ({self.pixel_size} entries)"
            )
        return in_features - self.pixel_size + self.latent_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        lead = x.shape[:-1]
        # Any number of leading axes, collapsed to one batch axis for the convs
        # and restored after.
        h = x[..., : self.pixel_size].reshape((-1,) + self.pixel_shape)
        for conv in self.conv_layers:
            h = self.activation_fn(as_float(conv(h)))
        h = as_float(self.projection(h.reshape(h.shape[0], -1)))
        # LayerNorm then tanh bounds the latent the MLP sees, the DrQ/SAC-AE
        # recipe, so a drifting encoder cannot blow up the head's input scale.
        h = jnp.tanh(self.norm(h)).reshape(lead + (self.latent_dim,))
        return jnp.concatenate([h, x[..., self.pixel_size:]], axis=-1)


def vision_encoder(
    vision: Optional[Mapping], in_features: int, *, rngs: nnx.Rngs
) -> tuple[Optional[PixelEncoder], int]:
    """Builds a network's encoder from its `vision` block.

    Returns:
        `(encoder, mlp_in_features)` — `(None, in_features)` when `vision` is
        `None`, so a network without one consumes no rngs and inits exactly as
        it did before vision existed.
    """
    if vision is None:
        return None, in_features
    encoder = PixelEncoder(**vision, rngs=rngs)
    return encoder, encoder.out_features(in_features)
