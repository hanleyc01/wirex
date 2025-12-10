"""The General Hebbian model implements the `Hebbian` class with a full Taylor expansion of
the Hebbian learning function.
"""

from typing import Self, override

import jax
from jaxtyping import Array, Float

from .hebbian import Coefficients, Hebbian


class GeneralHebbian(Hebbian):
    @classmethod
    def init(cls, pattern_dim: int, coefficients: Coefficients) -> Self: ...

    def fit(self, patterns: Float[Array, "N D"]) -> Self: ...

    @override
    @staticmethod
    def learning_rule(
        weights: jax.Array,
        coefficients: Coefficients,
        pre_synaptic_layer: jax.Array,
        post_synaptic_layer: jax.Array,
    ) -> jax.Array: ...
