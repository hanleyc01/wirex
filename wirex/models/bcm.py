"""[Bienenstock-Cooper-Munro Rule](https://neuronaldynamics.epfl.ch/online/Ch19.S2.html)"""

from typing import Self, override

from jaxtyping import Array, Float

from .coeffs import Coefficients
from .hebbian import Hebbian

__all__ = ["BCM"]


class BCM(Hebbian):
    """[Bienenstock-Cooper-Munro](https://neuronaldynamics.epfl.ch/online/Ch19.S2.html) associative memory model."""

    @classmethod
    def init(
        self, num_patterns: int, learning_rate: float, reference_rate: float
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
