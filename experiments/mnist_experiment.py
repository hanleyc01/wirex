"""Encapsulation of MNIST experimentation."""

import json
import sys
from dataclasses import dataclass
from typing import TextIO, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
from einops import rearrange
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

from experiments.similarity import cosine_similarity
from wirex.models.rate import Hopfield

__all__ = ["MnistExperiment", "PIXEL_WIDTH", "PIXEL_HEIGHT"]

_DATA_CACHE = "./.cache/mnist/"
"""Cache for serializing MNIST."""
PIXEL_WIDTH = 28
"""Pixel width of the MNIST dataset."""
PIXEL_HEIGHT = 28
"""Pixel height of the MNIST dataset."""


def _transform(data: torch.Tensor) -> NDArray[np.float32]:
    dataarr: NDArray[np.float32] = np.array(data, dtype=jnp.float32)
    dataarr = rearrange(dataarr, "w h -> (w h)")
    dataarr[dataarr > 0.0] = 1.0
    dataarr[dataarr == 0.0] = -1.0
    return dataarr


@dataclass
class MnistExperiment:
    """Encapuslation of experimentation and results into a class.

    Attributes:
        num_patterns int: The number of patterns
        train bool: Whether or not to sample from the train or test set.
        min int: The lower bound of patterns to store in the network.
        max int: The upper bound of patterns to store in the network.
        batch_size int: Batch size to sample from the dataloader.
        output (str | None): Defaults to `stdout`. Path to serialize output to.
    """

    train: bool
    min: int
    max: int
    batch_size: int
    output: str | None

    def run(self) -> None:
        print("[MNIST EXPERIMENT] BEGINNING MNIST EXPERIMENT", file=sys.stderr)
        results = []
        for i in tqdm(range(self.min, self.max)):
            mnist_train = MNIST(
                _DATA_CACHE, train=self.train, transform=_transform, download=True
            )
            mnist_data_loader = DataLoader(
                mnist_train, batch_size=self.batch_size, shuffle=True
            )
            mnist_it = iter(mnist_data_loader)
            mnist_data, _ = next(mnist_it)

            Xi = jnp.array(mnist_data[:i], dtype=jnp.float32)
            hopfield = Hopfield(Xi.T @ Xi)
            query = Xi[1]
            result = hopfield(query)
            results.append((i, query, result, cosine_similarity(query, result)))
        self.serialize_output(results)

    def serialize_output(
        self, results: list[tuple[int, jax.Array, jax.Array, jax.Array]]
    ) -> None:
        if not self.output:
            output = cast(TextIO, sys.stderr)
        else:
            output = open(self.output, "w")
        print(
            f"[MNIST EXPERIMENT] EXPERIMENT FINISHED, SERIALIZING RESULT TO {output}",
            file=sys.stderr,
        )
        result_dict = {}
        for idx, query, result, similarity in tqdm(results):
            result_dict["idx"] = idx
            result_dict["query"] = str(np.asarray(query).tobytes())
            result_dict["result"] = str(np.asarray(result).tobytes())
            result_dict["similarity"] = float(similarity)

        json.dump(result_dict, output)
        if output != sys.stdout:
            output.close()
