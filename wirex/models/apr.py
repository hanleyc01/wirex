"""Implementation of an [Activity Product Rule](https://neuronaldynamics.epfl.ch/online/Ch19.S2.html)."""

from typing import Self, override

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .coeffs import Coefficients
from .hebbian import Hebbian

__all__ = ["ActivityProductRule"]


class ActivityProductRule(Hebbian):
    r"""Activity Product Rule Hebbian associative memory.

    The activity product learning rule is defined as:
    $$
    \Delta W = \eta x x^\top,
    $$
    where $W$ is the weights of the model, $x$ is the pattern that
    you wish to memorize, and $\eta$ the learning rate.

    Example:
        `ActivityProductRule` is initializes with a the class method `ActivityProductRule.init`,
        and then fit to a list of patterns using `ActivityProductRule.fit`.
        ```
        patterns = ...
        pattern_dim = patterns.shape[-1]
        apr = ActivityProductRule.init(pattern_dim, learning_rate=1).fit(patterns)
        ```
    """

    weights: Float[Array, "D D"]
    coefficients: Coefficients

    @classmethod
    def init(cls, pattern_dim: int, learning_rate: float) -> Self:
        """Initialize a new `ActivityProductRule`.

        Args:
            pattern_dim: The dimensionality of the patterns that are to be stored.
            learning_rate: The learning rate used in in the learning rule.
        """
        weights: Float[Array, "D D"] = jnp.zeros(
            (pattern_dim, pattern_dim), dtype=jnp.float32
        )
        coefficients = Coefficients.init(c_2_corr=learning_rate)
        return cls(weights=weights, coefficients=coefficients)

    def fit(self, patterns: Float[Array, "N D"]) -> Self:
        """Fit an initialied `ActivityProductRule` model to a pattern set.

        Args:
            patterns:
                `(N, D)`-dimensional matrix of `N` patterns, where `D` is `pattern_dim` used in initialization.

        Returns:
            A new reference to an `ActivityProductRule` model with the weights updated
            according to `ActivityProductRule.learning_rule`.
        """

        def update_step(
            weights: Float[Array, "D D"], i: np.ndarray
        ) -> tuple[Float[Array, "D D"], None]:
            pattern = patterns[i]
            updated_weights = self.learning_rule(
                weights, self.coefficients, pattern, pattern
            )
            return updated_weights, None

        n = patterns.shape[0]

        final_weights, _ = lax.scan(update_step, init=self.weights, xs=np.arange(n))
        return self.__class__(weights=final_weights, coefficients=self.coefficients)

    @override
    @staticmethod
    def learning_rule(
        weights: Float[Array, "D D"],
        coefficients: Coefficients,
        pre_synaptic_layer: Float[Array, "D"],
        post_synaptic_layer: Float[Array, "D"],
    ) -> Float[Array, "D D"]:
        r"""Activity Product Rule learning rule:

        $$
        \Delta W = \eta x x^\top,
        $$
        """
        return weights + coefficients.c_2_corr * jnp.outer(
            pre_synaptic_layer, post_synaptic_layer
        )
