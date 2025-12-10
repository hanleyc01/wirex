"""`Bienenstock-Cooper-Munro Rule <https://neuronaldynamics.epfl.ch/online/Ch19.S2.html>`_"""

from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .hebbian import Coefficients, Hebbian


class BCM(Hebbian):
    @classmethod
    def init(
        cls, pattern_dim: int, learning_rate: float, activity_threshold: float
    ) -> Self:
        weights = jnp.zeros((pattern_dim, pattern_dim), dtype=jnp.float32)
        coefficients = Coefficients.init(
            c_3_pre=learning_rate, c_2_corr=-learning_rate * activity_threshold
        )
        return cls(weights=weights, coefficients=coefficients)

    def fit(self, patterns: Float[Array, "N D"]) -> Self: ...

    @override
    @staticmethod
    def learning_rule(
        weights: jax.Array,
        coefficients: Coefficients,
        pre_synaptic_layer: jax.Array,
        post_synaptic_layer: jax.Array,
    ) -> jax.Array: ...
