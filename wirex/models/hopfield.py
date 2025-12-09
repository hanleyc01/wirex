"""Hopfield network implementation."""

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["Hopfield"]


class Hopfield(eqx.Module):
    """Implementation of the [Hopfield network](https://en.wikipedia.org/wiki/Hopfield_network) using energy-descent.

    For more information about the mechanisms behind Hopfield networks, see
    this [tutorial on associative memories](https://tutorial.amemory.net/).

    Attributes:
        weights:
            `(D, D)`-dimensional matrix of full lateral connections between weights.
    """

    weights: Float[Array, "D D"]

    def recall(self, query: Float[Array, "D"]) -> Float[Array, "D"]:
        """Perform energy-descent minimization recall over `initial_state`.

        Args:
            query:
                `(D,)`-dimensional float vector for the initial state.

        Returns:
            The final recalled state.
        """
        return jnp.sign(self.weights.dot(query))
