"""Taylor coefficients used in Hebbian models."""

from typing import Self

import equinox as eqx

__all__ = ["Coefficients"]


class Coefficients(eqx.Module):
    """Taylor coefficients usedin Hebbian model learning rules. For more information,
    see `.hebbian.Hebbian` documentation.
    """

    c_0: float
    c_1_pre: float
    c_1_post: float
    c_2_pre: float
    c_2_post: float
    c_2_corr: float

    @classmethod
    def init(
        cls,
        c_0: float = 0.0,
        c_1_pre: float = 0.0,
        c_1_post: float = 0.0,
        c_2_pre: float = 0.0,
        c_2_post: float = 0.0,
        c_2_corr: float = 0.0,
    ) -> Self:
        return cls(
            c_0=c_0,
            c_1_pre=c_1_pre,
            c_1_post=c_1_post,
            c_2_pre=c_2_pre,
            c_2_post=c_2_post,
            c_2_corr=c_2_corr,
        )
