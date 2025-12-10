"""Generate random data."""

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float


def uniform(
    key: jax.Array, num_patterns: int, pattern_dim: int
) -> Float[Array, "num_patterns pattern_dim"]:
    """Generate uniformly sampled random data."""
    ...


def correlated(
    key: jax.Array, correlation: float, num_patterns: int, pattern_dim: int
) -> Float[Array, "num_patterns pattern_dim"]:
    """Generate correlated random data."""
    ...


def positive_skewed(
    key: jax.Array, skew: float, num_patterns: int, pattern_dim: int
) -> Float[Array, "num_patterns pattern_dim"]:
    """Generate positively skewed data, see Johns and Jones (2010)."""
    ...
