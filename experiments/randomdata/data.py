"""Generate random data."""

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

__all__ = ["uniform", "correlated", "positive_skewed"]


def uniform(
    key: jax.Array, num_patterns: int, pattern_dim: int
) -> Float[Array, "num_patterns pattern_dim"]:
    """Generate uniformly sampled random data.

    Args:
        key jax.Array: Random key to be used by `jax.random`.
        num_patterns int: The number of patterns to generate.
        pattern_dim int: The dimension of the patterns to generate.

    Returns:
        The uniformly distributed random matrix.
    """
    return jr.uniform(key, (num_patterns, pattern_dim), dtype=jnp.float32)


def correlated(
    key: jax.Array, correlation: float, num_patterns: int, pattern_dim: int
) -> Float[Array, "num_patterns pattern_dim"]:
    """Generate correlated random data."""
    ...


def positive_skewed(
    key: jax.Array, skew: float, num_patterns: int, pattern_dim: int
) -> Float[Array, "num_patterns pattern_dim"]:
    """Generate data with positive skew, see Johns and Jones (2010)."""
    ...
