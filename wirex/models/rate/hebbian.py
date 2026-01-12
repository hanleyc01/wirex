"""A Hebbian model is any associative memory which implements a *Hebbian* learning rule."""

from typing import Callable, Self

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["Hebbian", "Coefficients", "CoefficientType"]

CoefficientType = float | Callable[[jax.Array], jax.Array]
"""Type of coefficients: `float | Callable[[jax.Array], jax.Array]"""


class Coefficients(eqx.Module):
    c_0: CoefficientType
    c_1_pre: CoefficientType
    c_1_post: CoefficientType
    c_2_pre: CoefficientType
    c_2_post: CoefficientType
    c_2_corr: CoefficientType
    c_3_pre: CoefficientType
    c_3_post: CoefficientType

    @classmethod
    def init(
        cls,
        c_0: CoefficientType = 0.0,
        c_1_pre: CoefficientType = 0.0,
        c_1_post: CoefficientType = 0.0,
        c_2_pre: CoefficientType = 0.0,
        c_2_post: CoefficientType = 0.0,
        c_2_corr: CoefficientType = 0.0,
        c_3_pre: CoefficientType = 0.0,
        c_3_post: CoefficientType = 0.0,
    ) -> Self:
        return cls(
            c_0=c_0,
            c_1_pre=c_1_pre,
            c_1_post=c_1_post,
            c_2_pre=c_2_pre,
            c_2_post=c_2_post,
            c_2_corr=c_2_corr,
            c_3_pre=c_3_pre,
            c_3_post=c_3_post,
        )

    def jax(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        """Convert the `Coefficients` object into a `jax.Array`.

        Returns:
            Array with the format
            ```
            jax.array([self.c_0, self.c_1_pre, self.c_1_post, self.c_2_pre, self.c_2_post, self.c_2_corr, self.c_3_pre, self.c_3_post])
            ```
        """
        return jnp.array(
            [
                self.c_0,
                self.c_1_pre,
                self.c_1_post,
                self.c_2_pre,
                self.c_2_post,
                self.c_2_corr,
                self.c_3_pre,
                self.c_3_post,
            ],
            dtype=dtype,
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
    # TODO(hanleyc01): figure out a way to store the coefficients as an array rather
    # than as a reference to a new object. the goal is to make these things' computation
    # to be really fast.
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
