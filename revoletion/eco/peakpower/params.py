from dataclasses import dataclass

from revoletion.eco.params import CostParams


@dataclass(frozen=True)
class PeakPowerOpexParams(CostParams):
    spec: float | int = 0.0
    n_peak_periods_yr: int = 1
    n_peak_periods_sim: int = 1
