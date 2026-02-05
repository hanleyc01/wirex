"""Random models on uniformly generated data."""

from collections.abc import Iterator
from dataclasses import dataclass

import jax
import jax.random as jr

from wirex.models.rate import GeneralHebbian

from .experiment_result import ExperimentalResult
from .randomdata.models import random_models

KEY = jr.PRNGKey(111)


@dataclass
class UniformExperiment:
    """Encapsulation of the Uniform data experiment.

    Attributes:
        pattern_dim int:
            The dimensionality of the patterns to store.
        dtype str:
            The data type of the patterns.
        min int:
            The minimum number of patterns to store.
        max int:
            The maximum number of patterns to store (inclusive).
        output (str | None):
            Path to serialize the results to.
        num_models int:
            The number of models to generate.
    """

    pattern_dim: int
    dtype: str
    min: int
    max: int
    output: str | None
    num_models: int

    def generate_models(self) -> list[GeneralHebbian]:
        return random_models(
            key=KEY, num_models=self.num_models, pattern_dim=self.pattern_dim
        )

    def generate_data(self) -> Iterator[jax.Array]: ...

    def run(self) -> None: ...

    def serialize_output(self) -> None: ...
