"""A Hebbian model is any associative memory which implements a *Hebbian* learning rule."""

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["Hebbian", "Coefficients"]


class Coefficients(eqx.Module):
    c_0: float
    c_1_pre: float
    c_1_post: float
    c_2_pre: float
    c_2_post: float
    c_2_corr: float

    @classmethod
    def init(
        cls,
        c_0: float = 0.0,
        c_1_pre: float = 0.0,
        c_1_post: float = 0.0,
        c_2_pre: float = 0.0,
        c_2_post: float = 0.0,
        c_2_corr: float = 0.0,
    ) -> Self:
        return cls(
            c_0=c_0,
            c_1_pre=c_1_pre,
            c_1_post=c_1_post,
            c_2_pre=c_2_pre,
            c_2_post=c_2_post,
            c_2_corr=c_2_corr,
        )


class Hebbian(eqx.Module):
    """`Hebbian` Associative Memory is any associative memory which implements `Hebbian.learning_rule`.

    Attributes:
        weights:
            Matrix representing the weights of the Hebbian associative memory.
        corr:
            The Taylor coefficients for the Hebbian learning expansion (used in `self.learning_rule`).
    """

    weights: jax.Array
    coefficients: Coefficients

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
        ...

    def __call__(self, query: jax.Array) -> jax.Array:
        """Recall a stored pattern on the basis of a query.

        Args:
            query: Query to probe the associative memory.

        Returns:
            The stored pattern that the query excites.
        """
        return jnp.sign(self.weights.dot(query))
