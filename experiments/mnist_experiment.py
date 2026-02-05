"""Encapsulation of MNIST experimentation."""

import json
import sys
from dataclasses import asdict, dataclass
from typing import TextIO, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
from einops import rearrange
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

from experiments.similarity import cosine_similarity
from wirex.models.rate import GeneralHebbian

from .experiment_result import ExperimentalResult
from .randomdata.models import random_models

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
    num_models: int

    def generate_models(self) -> list[GeneralHebbian]:
        key = jr.PRNGKey(111)
        return random_models(
            key=key, num_models=self.num_models, pattern_dim=PIXEL_HEIGHT * PIXEL_WIDTH
        )

    def run(self) -> None:
        models = self.generate_models()
        print("[MNIST EXPERIMENT] BEGINNING MNIST EXPERIMENT", file=sys.stderr)

        results: list[ExperimentalResult] = []
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
            # hopfield = Hopfield(Xi.T @ Xi)
            # print("[MNIST EXPERIMENT] FITTING MODELS", file=sys.stderr)
            fitted_models = []
            for model in models:
                fitted_models.append(model.fit(Xi))

            query = Xi[1]

            # print("[MNIST EXPERIMENT] QUERYING MODELS", file=sys.stderr)
            for fitted_model in fitted_models:
                result = fitted_model(query)
                similarity = cosine_similarity(result, query)
                experimental_result = ExperimentalResult(
                    fitted_model.coefficients.tolist(),
                    i,
                    query.tolist(),
                    result.tolist(),
                    float(similarity),
                )
                results.append(experimental_result)

        self.serialize_output(results)

    def serialize_output(self, results: list[ExperimentalResult]) -> None:
        if not self.output:
            output = cast(TextIO, sys.stderr)
        else:
            output = open(self.output, "w")
        print(
            f"[MNIST EXPERIMENT] EXPERIMENT FINISHED, SERIALIZING RESULT TO {output}",
            file=sys.stderr,
        )

        data = [asdict(result) for result in tqdm(results)]
        # for idx, num_entries, query, result, similarity in tqdm(results):
        #     entry = {}
        #     entry["idx"] = idx
        #     entry["num_entries"] = num_entries
        #     entry["query"] = query.tolist()
        #     entry["result"] = result.tolist()
        #     entry["similarity"] = float(similarity)
        #     result_dict.append(entry)

        json.dump(data, output)
        if output != sys.stdout:
            output.close()
