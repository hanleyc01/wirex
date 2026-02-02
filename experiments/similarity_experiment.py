"""Random models on varying degrees of similar data, see also MNIST experiment."""

from dataclasses import dataclass


@dataclass
class SimilarityExperiment:
    """Encapsulation of similarity experimental condition."""

    def generate_models(self) -> None: ...

    def generate_data(self) -> None: ...

    def run(self) -> None: ...

    def serialize_results(self) -> None: ...
