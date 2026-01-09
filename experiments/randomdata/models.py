"""Generate random coefficients."""

import jax
import jax.numpy as jnp
import jax.random as jr

from wirex.models.rate import Coefficients, GeneralHebbian

__all__ = ["random_coefficients", "random_models"]


def random_coefficients(key: jax.Array, num_coefficients: int) -> list[Coefficients]:
    """Generate random coefficients.

     Args:
        key jax.Array: Random key used for generating coefficients.
        num_coefficients int: Number of coefficients to generate.

    Returns:
        List of coefficients with length equal to `num_coefficients`.
    """
    keys = jr.split(key, num_coefficients)
    coeffs: list[Coefficients] = []
    for ikey in keys:
        coeffarr = jr.uniform(key=ikey, shape=(8,), dtype=jnp.float32)
        coeffs.append(
            Coefficients(
                float(coeffarr[0]),
                float(coeffarr[1]),
                float(coeffarr[2]),
                float(coeffarr[3]),
                float(coeffarr[4]),
                float(coeffarr[5]),
                float(coeffarr[6]),
                float(coeffarr[7]),
            )
        )
    return coeffs


def random_models(
    key: jax.Array, num_models: int, pattern_dim: int
) -> list[GeneralHebbian]:
    """Generate random General Hebbian models.

    Uses the `random_coefficients` function to generate coefficients for
    each of the models, and then initializes new models.

    Args:
        key jax.Array: Random key used for generating coefficients.
        num_models: Number of models to generate.
        pattern_dim int: The dimensionality of the patterns.

    Returns:
        List of models with the length equal to `num_models`.
    """
    coeffs = random_coefficients(key, num_models)
    models = [GeneralHebbian.init(pattern_dim, coeff) for coeff in coeffs]
    return models
