"""`Bienenstock-Cooper-Munro Rule <https://neuronaldynamics.epfl.ch/online/Ch19.S2.html>`_"""

from typing import Self, override

import jax
from jaxtyping import Array, Float

from .hebbian import Coefficients, Hebbian


class BCM(Hebbian):
    @classmethod
    def init(
        cls, num_patterns: int, learning_rate: float, activity_threshold: float
    ) -> Self: ...

    def fit(self, patterns: Float[Array, "N D"]) -> Self: ...

    @override
    @staticmethod
    def learning_rule(
        weights: jax.Array,
        coefficients: Coefficients,
        pre_synaptic_layer: jax.Array,
        post_synaptic_layer: jax.Array,
    ) -> jax.Array: ...
