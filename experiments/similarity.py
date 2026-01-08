"""Collection of similarity functions for vectors.

For more information, see [https://en.wikipedia.org/wiki/Similarity_measure].
"""

import jax
import jax.numpy as jnp


def cosine_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Return the cosine similarity between two arrays."""

    x_norm = jnp.linalg.norm(x)
    x_norm = x_norm if x_norm != 0.0 else 0.1e-8
    y_norm = jnp.linalg.norm(y)
    y_norm = y_norm if y_norm != 0.0 else 0.1e-8

    return x.dot(y) / (x_norm * y_norm)


def dot_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Dot product similarity between two arrays."""

    return x.dot(y)


def hamming_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    """Hamming distance between two binary vectors."""
    ...


def chebyshev_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    """[Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance) function between
    two vectors.
    """
    ...


def levenshtein_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    """[Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
    between two vectors.
    """
    ...


def jaccard_index(x: jax.Array, y: jax.Array) -> jax.Array:
    """[Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) between two vectors."""
    ...
