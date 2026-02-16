from dataclasses import dataclass
from typing import Self

from revoletion.eco.params import CostParams


@dataclass(frozen=True)
class PeakPowerOpexParams(CostParams):
    spec: float
    n_peak_periods_yr: int
    n_peak_periods_sim: int

    @classmethod
    def create_from_plain(
        cls,
        spec: float | int,
        n_peak_periods_yr: int,
        n_peak_periods_sim: int,
        *args,
        **kwargs,
    ) -> Self:
        return cls(spec=spec, n_peak_periods_yr=n_peak_periods_yr, n_peak_periods_sim=n_peak_periods_sim)
