"""Implementation of Oja's rule."""

from typing import Callable, Self, cast, override

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .hebbian import Coefficients, Hebbian

__all__ = ["Oja"]


class Oja(Hebbian):
    """Implementation of `Oja's Rule <https://en.wikipedia.org/wiki/Oja%27s_rule>`_ as an associative memory."""

    @classmethod
    def init(cls, pattern_dim: int, gamma: float) -> Self:
        """Initialize a new Oja network.

        Raises:
            `ValueError`, if `gamma <= 0`.
        """
        if gamma <= 0:
            raise ValueError("Expected `gamma > 0`")
        weights = jnp.zeros((pattern_dim, pattern_dim), dtype=jnp.float32)
        coefficients = Coefficients.init(c_2_post=lambda w: -gamma * w, c_2_corr=gamma)
        return cls(weights=weights, coefficients=coefficients)

    def fit(self, patterns: Float[Array, "N D"]) -> Self: ...

    @override
    @staticmethod
    def learning_rule(
        weights: jax.Array,
        coefficients: Coefficients,
        pre_synaptic_layer: jax.Array,
        post_synaptic_layer: jax.Array,
    ) -> jax.Array:
        c_2_corr = cast(float, coefficients.c_2_corr)
        c_2_post = cast(Callable[[jax.Array], jax.Array], coefficients.c_2_post)
        return (
            weights
            + c_2_corr * jnp.outer(pre_synaptic_layer, post_synaptic_layer)
            + c_2_post(weights) * post_synaptic_layer
        )
