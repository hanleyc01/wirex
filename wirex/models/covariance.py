"""[Covariance learning rule](https://neuronaldynamics.epfl.ch/online/Ch19.S2.html)."""

from typing import Self, override

from jaxtyping import Array, Float

from .hebbian import Coefficients, Hebbian


class Covariance(Hebbian):
    @classmethod
    def init(
        cls,
        num_patterns: int,
        expected_presynaptic_rate: float,
        expected_postsynaptic_rate: float,
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
