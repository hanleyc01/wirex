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

    equals_zero = x == y
    return jnp.sum(1 - equals_zero)


def chebyshev_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    """[Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance) function between
    two vectors.
    """
    diff = x - y
    return jnp.max(diff)


def levenshtein_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    """[Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
    between two vectors.
    """

    m = len(x)
    n = len(y)

    prev = jnp.arange(n + 1, dtype=int)
    curr = jnp.zeros(n + 1, dtype=int)

    for i in range(1, m + 1):
        curr = curr.at[0].set(i)
        for j in range(1, n + 1):
            if x[i - 1] * y[i - 1] == 1:
                cost = 0
            else:
                cost = 1
            curr = curr.at[j].set(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev, curr = curr, prev

    return prev[n]


def jaccard_index(x: jax.Array, y: jax.Array) -> jax.Array:
    """[Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) between two vectors."""
    ...
