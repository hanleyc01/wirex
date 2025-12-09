"""Covariance Rule."""

from typing import Self, override

from jaxtyping import Array, Float

from .coeffs import Coefficients
from .hebbian import Hebbian

__all__ = ["Covariance"]


class Covariance(Hebbian):
    """Covariance rule associative memory."""

    @classmethod
    def init(
        self,
        num_patterns: int,
        pre_synaptic_expected_value: Float[Array, "D"],
        post_synaptic_expected_value: Float[Array, "D"],
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
