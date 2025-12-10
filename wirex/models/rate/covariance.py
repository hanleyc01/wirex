"""`Covariance learning rule <https://neuronaldynamics.epfl.ch/online/Ch19.S2.html>`_."""

from typing import Self, cast, override

import jax.numpy as jnp
from jaxtyping import Array, Float

from .hebbian import Coefficients, Hebbian


class Covariance(Hebbian):
    r"""Covariance learning rule, where rates fluctuate around mean expected values.

    Let $\langle x_i\rangle$ and $\langle y_j \rangle$ be the expected value of
    the elements of the pre-synaptic layer $x$ and the post-synaptic layer $y$.
    Let $W$ be the weights between the layers. Then, the coviariance learning
    rule is:

    .. math::
        \Delta W_{ij} = \gamma (x_i - \langle x_i \rangle) (y_j - \langle y_j \rangle).

    Training a covariance network requires getting the difference between
    the post- and pre-synaptic rates with their expected values. To compute and use
    this value in training, use the `Covariance.presynaptic` and `Covariance.postsynaptic`
    methods.

    Attributes:
        weights: The dimensionality of the patterns stored.
        coefficients: The Hebbian coefficients.
        expected_presynaptic_rates: Expected presynaptic neuron rates.
        expected_postsynaptic_rates: Expected postsynaptic neuron rates.
    """

    weights: Float[Array, "D D"]
    coefficients: Coefficients
    expected_presynaptic_rates: Float[Array, "D"]
    expected_postsynaptic_rates: Float[Array, "D"]

    @classmethod
    def init(
        cls,
        pattern_dim: int,
        learning_rate: float,
        expected_presynaptic_rates: Float[Array, "D"],
        expected_postsynaptic_rates: Float[Array, "D"],
    ) -> Self:
        """Initialize a new covariance model.

        Args:
            pattern_dim: The dimensionality of the patterns stored.
            learning_rate: The learning rate of the neural network.
            expected_presynaptic_rates: Expected presynaptic neuron rates.
            expected_postsynaptic_rates: Expected postsynaptic neuron rates.

        Returns:
            `Self`.
        """
        return cls(
            weights=jnp.zeros((pattern_dim, pattern_dim), dtype=jnp.float32),
            coefficients=Coefficients.init(c_2_corr=learning_rate),
            expected_postsynaptic_rates=expected_postsynaptic_rates,
            expected_presynaptic_rates=expected_presynaptic_rates,
        )

    def fit(self, patterns: Float[Array, "N D"], num_iters: int) -> Self: ...

    def presynaptic(self, presynaptic_rate: Float[Array, "D"]) -> Float[Array, "D"]:
        """The difference of the presynaptic rate with its expected value."""
        return presynaptic_rate - self.expected_presynaptic_rates

    def postsynaptic(self, postsynaptic_rate: Float[Array, "D"]) -> Float[Array, "D"]:
        """The difference of the postsynaptic rate with its expected value."""
        return postsynaptic_rate - self.expected_postsynaptic_rates

    @override
    @staticmethod
    def learning_rule(
        weights: Float[Array, "D D"],
        coefficients: Coefficients,
        pre_synaptic_layer: Float[Array, "D"],
        post_synaptic_layer: Float[Array, "D"],
    ) -> Float[Array, "D D"]:
        r"""Covariance learning rule.


        .. math::
            \Delta W_{ij} = \gamma (x_i - \langle x_i \rangle) (y_j - \langle y_j \rangle).
        """
        return weights + cast(float, coefficients.c_2_corr) * jnp.outer(
            pre_synaptic_layer, post_synaptic_layer
        )
