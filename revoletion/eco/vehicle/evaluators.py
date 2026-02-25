from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Type, Self

import numpy as np
import pandas as pd

from revoletion.eco import EcoParams, Evaluator
from revoletion.eco.evaluators import OpexEvaluator, CrevEvaluator
from revoletion.eco.utils import transform_scalar_var

from .params import VehicleOpexParams, VehicleCrevParams


class VehicleOpexEvaluator(OpexEvaluator):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        spec: pd.Series,
        fix: float,
        spec_dist: pd.Series,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            spec=spec,
            fix=fix,
            **kwargs,
        )
        self.spec_dist = spec_dist

    @classmethod
    def _build_kwargs_from_params(cls, params: VehicleOpexParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict:
        return {
            **super()._build_kwargs_from_params(params=params, eco=eco, data_dir=data_dir, **kwargs),
            "spec_dist": transform_scalar_var(value=params.spec_dist, dti=eco.dti_eval, data_dir=data_dir),
        }

    def _calc_eval(self, flow: pd.Series | None, dist: pd.Series | None = None, **kwargs) -> float:
        cost_dist = (np.dot(self.spec_dist.to_numpy(), dist[self.eco.dti_eval].to_numpy())) if dist is not None else 0.0
        return super()._calc_eval(flow=flow, **kwargs) + cost_dist

    def evaluate(self, flow: pd.Series, dist: pd.Series | None = None, **kwargs):
        super().evaluate(flow=flow, dist=dist, **kwargs)


class VehicleCrevEvaluator(CrevEvaluator):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        spec: pd.Series,
        fix: float,
        spec_dist: pd.Series,
        spec_time: pd.Series,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            spec=spec,
            fix=fix,
            **kwargs,
        )

        self.spec_dist = spec_dist
        self.spec_time = spec_time

    @classmethod
    def _build_kwargs_from_params(cls, params: VehicleCrevParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict:
        return {
            **super()._build_kwargs_from_params(params=params, eco=eco, data_dir=data_dir, **kwargs),
            "spec_dist": transform_scalar_var(value=params.spec_dist, dti=eco.dti_eval, data_dir=data_dir),
            "spec_time": transform_scalar_var(value=params.spec_time, dti=eco.dti_eval, data_dir=data_dir),
        }

    def _calc_eval(
        self, flow: pd.Series | None, dist: pd.Series | None = None, atbase: pd.Series | None = None, **kwargs
    ) -> float:
        crev_dist = (np.dot(self.spec_dist.to_numpy(), dist[self.eco.dti_eval].to_numpy())) if dist is not None else 0.0
        crev_time = (
            (np.dot(self.spec_time.to_numpy(), (~atbase[self.eco.dti_eval].astype(bool).to_numpy()).astype(int)))
            if atbase is not None
            else 0.0
        )
        return super()._calc_eval(flow=flow, **kwargs) + crev_dist + crev_time

    def evaluate(self, flow: pd.Series, dist: pd.Series | None = None, atbase: pd.Series | None = None, **kwargs):
        super().evaluate(flow=flow, dist=dist, atbase=atbase, **kwargs)


@dataclass
class VehicleEvaluator(Evaluator):
    """
    EvaluatorBlock for vehicles, which includes specific methods to evaluate vehicle-specific costs and results.
    """

    OPEX_EVALUATOR: ClassVar[Type[OpexEvaluator]] = VehicleOpexEvaluator
    CREV_EVALUATOR: ClassVar[Type[CrevEvaluator]] = VehicleCrevEvaluator

    def evaluate(
        self,
        size_preexisting: float | None = None,
        size_expansion: float | None = None,
        flow: pd.Series | None = None,
        dist: pd.Series | None = None,
        atbase: pd.Series | None = None,
        **kwargs,
    ) -> None:
        super().evaluate(
            size_preexisting=size_preexisting,
            size_expansion=size_expansion,
            flow=flow,
            dist=dist,
            atbase=atbase,
            **kwargs,
        )
