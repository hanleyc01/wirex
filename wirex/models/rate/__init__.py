"""Rate-based Hebbian associative memory models."""

from .apr import ActivityProductRule
from .bcm import BCM
from .covariance import Covariance
from .generalhebbian import GeneralHebbian
from .hebbian import Coefficients, Hebbian
from .hopfield import Hopfield
from .oja import Oja

__all__ = [
    "ActivityProductRule",
    "BCM",
    "Covariance",
    "GeneralHebbian",
    "Coefficients",
    "Hebbian",
    "Hopfield",
    "Oja",
]
