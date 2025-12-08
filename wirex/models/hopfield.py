"""Hopfield network implementation."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped


class Hopfield(eqx.Module):
    """Implementation of the [Hopfield network](https://en.wikipedia.org/wiki/Hopfield_network)
    using energy-descent.

    For more information about the mechanisms behind Hopfield networks, see
    this [tutorial on associative memories](https://tutorial.amemory.net/).

    Attributes:
        weights (Float[Array, "D D"]):
            `(D, D)`-dimensional matrix of full lateral connections between weights.
    """

    weights: Float[Array, "D D"]

    @jaxtyped(typechecker=beartype)
    def recall(self, initial_state: Float[Array, "D"]) -> Float[Array, "D"]:
        """Perform energy-descent minimization recall over `initial_state`.

        Args:
            initial_state (Float[Array, "D"]):
                `(D,)`-dimensional float vector for the initial state.

        Returns:
            The final recalled state.
        """
        return jnp.sign(self.weights.dot(initial_state))
