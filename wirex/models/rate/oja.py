"""Implementation of Oja's rule."""

from typing import Self, override

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .hebbian import Coefficients, Hebbian

__all__ = ["Oja"]


class Oja(Hebbian):
    """Implementation of `Oja's Rule <https://en.wikipedia.org/wiki/Oja%27s_rule>`_ as an associative memory."""

    @classmethod
    def init(cls, pattern_dim: int, gamma: float) -> Self: ...

    def fit(self, patterns: Float[Array, "N D"]) -> Self: ...

    @override
    @staticmethod
    def learning_rule(
        weights: jax.Array,
        coefficients: Coefficients,
        pre_synaptic_layer: jax.Array,
        post_synaptic_layer: jax.Array,
    ) -> jax.Array: ...
