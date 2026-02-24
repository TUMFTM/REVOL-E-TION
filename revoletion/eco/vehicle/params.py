from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pandas as pd

from revoletion.eco.params import OpexParams, CrevParams
from revoletion.eco.utils import transform_scalar_var


@dataclass(frozen=True)
class VehicleOpexParams(OpexParams):
    spec_dist: pd.Series

    @classmethod
    def create(
        cls,
        dti_sim: pd.DatetimeIndex,
        spec: str | float | int = 0.0,
        fix: float | int = 0.0,
        data_dir: Path = None,
        *args,
        **kwargs,
    ) -> Self:
        spec_dist = kwargs.get("spec_dist", None)
        if spec_dist is None:
            raise ValueError("VehicleOpexParams requires a 'spec_dist' argument")
        spec_series = transform_scalar_var(value=spec, dti=dti_sim, data_dir=data_dir)
        spec_dist_series = transform_scalar_var(value=spec_dist, dti=dti_sim, data_dir=data_dir)
        return cls(spec=spec_series, fix=fix, spec_dist=spec_dist_series)


@dataclass(frozen=True)
class VehicleCrevParams(CrevParams):
    spec_dist: pd.Series
    spec_time: pd.Series

    @classmethod
    def create(
        cls,
        dti_sim: pd.DatetimeIndex,
        spec: str | float | int = 0.0,
        fix: float | int = 0.0,
        data_dir: Path = None,
        *args,
        **kwargs,
    ) -> Self:
        spec_dist = kwargs.get("spec_dist", None)
        if spec_dist is None:
            raise ValueError("VehicleOpexParams requires a 'spec_dist' argument")
        spec_time = kwargs.get("spec_time", None)
        if spec_time is None:
            raise ValueError("VehicleOpexParams requires a 'spec_time' argument")
        spec_series = transform_scalar_var(value=spec, dti=dti_sim, data_dir=data_dir)
        spec_dist_series = transform_scalar_var(value=spec_dist, dti=dti_sim, data_dir=data_dir)
        spec_time_series = transform_scalar_var(value=spec_time, dti=dti_sim, data_dir=data_dir)
        return cls(spec=spec_series, fix=fix, spec_dist=spec_dist_series, spec_time=spec_time_series)
