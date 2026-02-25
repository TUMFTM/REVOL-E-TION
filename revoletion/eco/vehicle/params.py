from dataclasses import dataclass

from revoletion.eco.params import OpexParams, CrevParams


@dataclass(frozen=True)
class VehicleOpexParams(OpexParams):
    spec_dist: str | float | int = 0.0


@dataclass(frozen=True)
class VehicleCrevParams(CrevParams):
    spec_dist: str | float | int = 0.0
    spec_time: str | float | int = 0.0
