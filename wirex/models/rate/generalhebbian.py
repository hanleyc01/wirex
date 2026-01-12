"""The General Hebbian model implements the `Hebbian` class with a full Taylor expansion of
the Hebbian learning function.
"""

from typing import Self, override

import jax
import jax.numpy as jnp
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
        return cls(
            weights=jnp.zeros((pattern_dim, pattern_dim), dtype=jnp.float32),
            coefficients=coefficients,
        )

    def fit(self, patterns: Float[Array, "N D"]) -> Self: ...

    @override
    @staticmethod
    def learning_rule(
        weights: jax.Array,
        coefficients: Coefficients,
        pre_synaptic_layer: jax.Array,
        post_synaptic_layer: jax.Array,
    ) -> jax.Array:
        r"""A Hebbian learning rule updates `self.weights` in terms of local pre-"synaptic" and
        post-"synaptic" layers.

        Using the indices of the weights of the matrix, a Hebbian learning rule is of
        the form:

        .. math::
            W_{ij} = c_0 + c_1^\text{pre} x_i + c_2^\text{post} y_j + c_2^\text{pre} x_i^2 + c_2^\text{post} y_j^2 + c_2^\text{corr} x_i y_j.

        Args:
            weights:
                The model's weights.
            coefficients:
                The coefficients of the model.
            pre_synaptic_layer:
                The pre-"synaptic" layer to be used in updating the weights.
            post_synaptic_layer:
                The post-"synaptic" layer to be used in updating the weights.

        Returns:
            The updated weights in terms of the models correlation coefficients
            and pre- and post-"synaptic" layers.
        """
        # TODO(hanleyc01): optimize this loop to use `lax` builtins for speed-up; see note in `Self.init`, but
        # also note how we use convert to an array here: wouldn't it be easier to store an array *already*?
        coeffsarr = coefficients.jax()
        pattern_dim, _ = weights.shape
        new_weights = jnp.zeros_like(weights)
        for i in range(pattern_dim):
            for j in range(pattern_dim):
                weights_ij = new_weights.at[i, j].add(
                    coeffsarr[0]
                    + coeffsarr[1] * pre_synaptic_layer[i]
                    + coeffsarr[2] * post_synaptic_layer[j]
                    + coeffsarr[3] * (pre_synaptic_layer[i] ** 2)
                    + coeffsarr[4] * (post_synaptic_layer**2)
                    + coeffsarr[5] * pre_synaptic_layer[i] * post_synaptic_layer[j]
                    + coeffsarr[6] * (pre_synaptic_layer**3)
                    + coeffsarr[7] * (post_synaptic_layer**3)
                )
                new_weights = new_weights.at[i, j].set(weights_ij)
        return new_weights
