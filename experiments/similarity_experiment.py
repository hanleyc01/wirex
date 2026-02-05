"""Random models on varying degrees of similar data, see also MNIST experiment."""

from dataclasses import dataclass

import jax
import jax.random as jr

from wirex.models.rate import GeneralHebbian

from .randomdata.data import correlated
from .randomdata.models import random_models

KEY = jr.PRNGKey(111)


@dataclass
class SimilarityExperiment:
    """Encapsulation of similarity experimental condition.

    Attributes:
        similarity float: Cosine similarity of data to generate.
        pattern_dim int: Dimensionality of the patterns to store.
        min int: The minimum number of patterns to store.
        max int: The maximum number of patterns to store.
        output (str | None): Path to serialize results to.
        num_models int: The number of models to generate.
    """

    similarity: float
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

    def generate_data(self) -> None: ...

    def run(self) -> None: ...

    def serialize_results(self) -> None: ...
