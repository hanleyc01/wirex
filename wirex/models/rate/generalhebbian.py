"""The General Hebbian model implements the `Hebbian` class with a full Taylor expansion of
the Hebbian learning function.
"""

from typing import Self, override

import jax
from jaxtyping import Array, Float

from .hebbian import Coefficients, Hebbian


class GeneralHebbian(Hebbian):
    """The `GeneralHebbian` model implements the `Hebbian` class, and uses
    the full Taylor expansion of the Hebbian learning function in its
    learning rule definition.
    """

    @classmethod
    def init(cls, pattern_dim: int, coefficients: Coefficients) -> Self:
        """Initialize a new `GeneralHebbian` model.

        Args:
            pattern_dim int: The dimensionality of the patterns.
            coefficients: The coefficients to use in the model.

        Returns:
            A new `GeneralHebbian` model.
        """
        # TODO(hanleyc01): this might just a future optimization before
        # running the final testing, but think about a way to make it so that
        # we can store arrays rather than actual coefficients?
        coeffs = coefficients.jax()
        ...

    def fit(self, patterns: Float[Array, "N D"]) -> Self: ...

    @override
    @staticmethod
    def learning_rule(
        weights: jax.Array,
        coefficients: Coefficients,
        pre_synaptic_layer: jax.Array,
        post_synaptic_layer: jax.Array,
    ) -> jax.Array: ...
