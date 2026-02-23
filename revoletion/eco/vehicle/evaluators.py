from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Type, Self

import numpy as np
import pandas as pd

from revoletion.eco import EcoParams, Evaluator
from revoletion.eco.evaluators import OpexEvaluator, CrevEvaluator
from revoletion.eco.vehicle.params import VehicleOpexParams, VehicleCrevParams


class VehicleOpexEvaluator(OpexEvaluator):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: VehicleOpexParams,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
            **kwargs,
        )

    def _calc_eval(self, flow: pd.Series, **kwargs) -> float:
        dist = kwargs.pop("dist", None)
        if dist is None:
            raise ValueError("VehicleOpexEvaluator requires a 'dist' argument")
        return (
            # energy based opex
            super()._calc_eval(flow=flow, **kwargs)
            # distance based opex
            + np.dot(self.params.spec_dist.to_numpy(), dist[self.eco.dti_eval].to_numpy())
        )

    def evaluate(self, flow: pd.Series, **kwargs):
        dist = kwargs.pop("dist", None)
        if dist is None:
            raise ValueError("VehicleOpexEvaluator requires a 'dist' argument")
        super().evaluate(flow=flow, dist=dist, **kwargs)


class VehicleCrevEvaluator(CrevEvaluator):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: VehicleCrevParams,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
            **kwargs,
        )

    def _calc_eval(self, flow: pd.Series, **kwargs) -> float:
        dist = kwargs.pop("dist", None)
        if dist is None:
            raise ValueError("VehicleOpexEvaluator requires a 'dist' argument")
        atbase = kwargs.pop("atbase", None)
        if atbase is None:
            raise ValueError("VehicleOpexEvaluator requires a 'atbase' argument")
        return (
            # energy based crev
            super()._calc_eval(flow=flow, **kwargs)
            # distance based crev
            + np.dot(self.params.spec_dist.to_numpy(), dist[self.eco.dti_eval].to_numpy())
            # time based crev (only for non-atbase periods)
            + np.dot(self.params.spec_time.to_numpy(), (~atbase[self.eco.dti_eval].astype(bool).to_numpy()).astype(int))
        )

    def evaluate(self, flow: pd.Series, **kwargs):
        dist = kwargs.pop("dist", None)
        if dist is None:
            raise ValueError("VehicleOpexEvaluator requires a 'dist' argument")
        atbase = kwargs.pop("atbase", None)
        if atbase is None:
            raise ValueError("VehicleOpexEvaluator requires a 'atbase' argument")
        super().evaluate(flow=flow, dist=dist, atbase=atbase, **kwargs)


@dataclass
class VehicleEvaluator(Evaluator):
    """
    EvaluatorBlock for vehicles, which includes specific methods to evaluate vehicle-specific costs and results.
    """

    OPEX_EVALUATOR: ClassVar[Type[OpexEvaluator]] = VehicleOpexEvaluator
    CREV_EVALUATOR: ClassVar[Type[CrevEvaluator]] = VehicleCrevEvaluator

    @classmethod
    def _build_opex_params(
        cls,
        eco: EcoParams,
        data_dir: Path,
        spec: float | Path,
        fix: float,
        **kwargs,
    ) -> VehicleOpexParams:
        spec_dist = kwargs.get("spec_dist", None)
        if spec_dist is None:
            raise ValueError("VehicleOpexParams requires a 'spec_dist' argument")
        return VehicleOpexParams.create_from_plain(
            spec=spec,
            spec_dist=spec_dist,
            fix=fix,
            dti_sim=eco.dti_sim,
            data_dir=data_dir,
        )

    @classmethod
    def _build_crev_params(
        cls,
        eco: EcoParams,
        data_dir: Path,
        spec: float | Path,
        fix: float,
        **kwargs,
    ) -> VehicleCrevParams:
        spec_dist = kwargs.get("spec_dist", None)
        if spec_dist is None:
            raise ValueError("VehicleOpexParams requires a 'spec_dist' argument")
        spec_time = kwargs.get("spec_time", None)
        if spec_time is None:
            raise ValueError("VehicleOpexParams requires a 'spec_time' argument")
        return VehicleCrevParams.create_from_plain(
            spec=spec,
            fix=fix,
            spec_dist=spec_dist,
            spec_time=spec_time,
            dti_sim=eco.dti_sim,
            data_dir=data_dir,
        )

    @classmethod
    def create_from_plain(
        cls,
        name: str,
        eco: EcoParams,
        data_dir: Path,
        name_size: str = None,
        name_flow: str = None,
        # capex
        capex_spec: float = 0.0,
        capex_fix: float = 0.0,
        capex_ccr: float = 1.0,
        consider_preexisting: bool = True,
        ls: int | None = None,
        age_preexisting: int = 0,
        capex_residual_at_ls: float = 0.0,
        # mntex
        mntex_spec: float = 0.0,
        mntex_fix: float = 0.0,
        # opex
        opex_spec: float | Path = 0.0,
        opex_spec_dist: float | Path = 0.0,
        opex_fix: float = 0.0,
        # crev
        crev_spec: float | Path = 0.0,
        crev_spec_dist: float | Path = 0.0,
        crev_spec_time: float | Path = 0.0,
        crev_fix: float = 0.0,
    ) -> Self:
        params_capex = cls._build_capex_params(
            eco=eco,
            spec=capex_spec,
            fix=capex_fix,
            ccr=capex_ccr,
            consider_preexisting=consider_preexisting,
            ls=ls,
            age_preexisting=age_preexisting,
            residual_at_ls=capex_residual_at_ls,
        )

        params_mntex = cls._build_mntex_params(
            spec=mntex_spec,
            fix=mntex_fix,
        )

        params_opex = cls._build_opex_params(
            eco=eco,
            data_dir=data_dir,
            spec=opex_spec,
            spec_dist=opex_spec_dist,
            fix=opex_fix,
        )

        params_crev = cls._build_crev_params(
            eco=eco,
            data_dir=data_dir,
            spec=crev_spec,
            spec_dist=crev_spec_dist,
            spec_time=crev_spec_time,
            fix=crev_fix,
        )

        return cls.create(
            name=name,
            eco=eco,
            name_size=name_size,
            name_flow=name_flow,
            params_capex=params_capex,
            params_mntex=params_mntex,
            params_opex=params_opex,
            params_crev=params_crev,
        )

    def evaluate(
        self,
        size_preexisting: float | None = None,
        size_expansion: float | None = None,
        flow: pd.Series | None = None,
        **kwargs,
    ) -> None:
        # ToDo: Explicitly add dist and atbase to the method signature
        dist = kwargs.get("dist", None)
        if dist is None:
            raise ValueError("VehicleEvaluator requires a 'dist' argument")
        atbase = kwargs.get("atbase", None)
        if atbase is None:
            raise ValueError("VehicleEvaluator requires a 'atbase' argument")

        super().evaluate(size_preexisting=size_preexisting, size_expansion=size_expansion, flow=flow, **kwargs)
