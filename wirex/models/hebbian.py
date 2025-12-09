"""A Hebbian model is any associative memory which implements a *Hebbian* learning rule."""

import equinox as eqx
import jax

__all__ = ["Hebbian"]


class Hebbian(eqx.Module):
    """`Hebbian` Associative Memory is any associative memory which implements `Hebbian.learning_rule`.

    Attributes:
        weights:
            Matrix representing the weights of the Hebbian associative memory.
        corr:
            The Taylor coefficients for the Hebbian learning expansion (used in `self.learning_rule`).
    """

    weights: jax.Array
    corr: jax.Array

    @staticmethod
    def learning_rule(
        weights: jax.Array,
        coefficients: jax.Array,
        pre_synaptic_layer: jax.Array,
        post_synaptic_layer: jax.Array,
    ) -> jax.Array:
        r"""A Hebbian learning rule updates `self.weights` in terms of local pre-"synaptic" and
        post-"synaptic" layers.

        Using the indices of the weights of the matrix, a Hebbian learning rule is of
        the form:
        $$
        W_{ij} = c_0 + c_1^\text{pre} x_i + c_2^\text{post} y_j + c_2^\text{pre} x_i^2 + c_2^\text{post} y_j^2 + c_2^\text{corr} x_i y_j.
        $$

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

    def recall(self, query: jax.Array) -> jax.Array:
        """Recall a stored pattern on the basis of a query.

        Args:
            query: Query to probe the associative memory.

        Returns:
            The stored pattern that the query excites.
        """
        ...
