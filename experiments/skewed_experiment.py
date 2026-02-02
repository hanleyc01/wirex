"""Random models on skewed data."""

from dataclasses import dataclass


@dataclass
class SkewedExperiment:
    """Encapsulation of skewed experimental condition."""

    def generate_models(self) -> None: ...

    def generate_data(self) -> None: ...

    def run(self) -> None: ...

    def serialize_results(self) -> None: ...
