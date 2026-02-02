from dataclasses import dataclass


@dataclass
class ExperimentalResult:
    """Dataclass to capture the results of an experimental trial.

    Attributes:
        coeffs: The coefficients of the model.
        patterns_stored: The number of patterns stored.
        query: The query pattern.
        result: The result pattern.
        cosine_similarity: The cosine similarity between the query and the result.
    """

    coeffs: list[float]
    patterns_stored: int
    query: list[float]
    result: list[float]
    cosine_similarity: float
