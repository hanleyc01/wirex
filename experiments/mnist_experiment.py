"""Encapsulation of MNIST experimentation."""

from dataclasses import dataclass

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
from einops import rearrange
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

__all__ = ["MnistExperiment"]

_DATA_CACHE = "./.cache/mnist/"
"""Cache for serializing MNIST."""
_PIXEL_WIDTH = 28
"""Pixel width of the MNIST dataset."""
_PIXEL_HEIGHT = 28
"""Pixel height of the MNIST dataset."""


def transform(data: torch.Tensor) -> NDArray[np.float32]:
    dataarr: NDArray[np.float32] = np.array(data, dtype=jnp.float32)
    dataarr = rearrange(dataarr, "w h -> (w h)")
    dataarr[dataarr > 0.0] = 1.0
    dataarr[dataarr == 0.0] = -1.0
    return dataarr


@dataclass
class MnistExperiment:
    num_patterns: int
    train: bool
    min: int
    max: int
    batch_size: int

    def run(self) -> None:
        dim = _PIXEL_HEIGHT * _PIXEL_WIDTH
        num_patterns = self.num_patterns

        for i in range(self.min, self.max):
            mnist_train = MNIST(
                _DATA_CACHE, train=self.train, transform=transform, download=True
            )
            mnist_data_loader = DataLoader(
                mnist_train, batch_size=self.batch_size, shuffle=True
            )
            mnist_it = iter(mnist_data_loader)
            mnist_data, _ = next(mnist_it)

            Xi = jnp.array(mnist_data[:2], dtype=jnp.float32)
            hopfield = Hopfield(Xi.T @ Xi)
            query = Xi[1]
            result = hopfield(query)
