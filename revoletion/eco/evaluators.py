from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self, Type

import numpy as np
import pandas as pd
from numpy import typing as npt

from .abstractclasses import (
    CalculableBaseElement,
    CalculableYearlyElement,
    CalculablePowerBasedElement,
    CapexElement,
    MntexElement,
    OpexElement,
    CrevElement,
    BlockElement,
)
from .params import EcoParams, CostParams, CapexParams, MntexParams, PowerBasedParams, OpexParams, CrevParams

from .utils import (
    OccursAt,
    calc_residual_value,
    calc_lifetime_remaining,
    DEPRECIATION,
)


class CostEvaluator(CalculableBaseElement, ABC):
    """
    Base class for all Evaluators that evaluate costs, i.e. Capex, Mntex, Opex and Crev.
    """

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: CostParams,
        **kwargs,
    ):
        super().__init__(name=name, eco=eco, **kwargs)
        self.eco = eco
        self.params = params

    @abstractmethod
    def _calc_spec_ep(self, **kwargs) -> float | pd.Series: ...

    @property
    def spec_ep(self) -> float | pd.Series:
        return self._calc_spec_ep()

    @property
    def spec(self) -> float | pd.Series:
        return self.params.spec

    @property
    def fix(self) -> float:
        return self.params.fix


class PowerBasedEvaluator(CostEvaluator, CalculablePowerBasedElement, ABC):
    _OCCURS_AT = OccursAt.END

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: PowerBasedParams,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
            **kwargs,
        )

    def _calc_eval(self, flow: pd.Series | None, **kwargs) -> float:
        cost_flow = (
            np.dot(self.params.spec.to_numpy(), flow[self.eco.dti_eval].to_numpy()) * self.eco.timestep_hours
            if flow is not None
            else 0.0
        )
        return cost_flow + self.params.fix

    def evaluate(self, flow: pd.Series | None, **kwargs):
        super().evaluate(flow=flow, **kwargs)

    def _calc_spec_ep(self, **kwargs) -> pd.Series:
        # calculate annuity due factor to compensate operation costs for difference between simulation and project time
        factor_operation_ep = (1 / self.eco.sim_yr_rat) if self.eco.compensate_sim_prj else 1

        return self.params.spec * factor_operation_ep


class CapexEvaluator(CostEvaluator, CapexElement):
    _OCCURS_AT = OccursAt.BEGIN

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: CapexParams,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
            **kwargs,
        )

    def _calc_preexisting(self, size_preexisting: float | None) -> float:
        cost_size = size_preexisting * self.params.spec if size_preexisting else 0.0
        return (cost_size + self.params.fix) * int(self.params.consider_preexisting)

    def _calc_expansion(self, size_expansion: float | None) -> float:
        return size_expansion * self.params.spec if size_expansion else 0.0

    def _calc_init(self) -> float:
        return self.preexisting + self.expansion

    def _calc_cashflow_factors(self, invest_first: int, **kwargs) -> npt.NDArray:
        invest_periods = np.arange(invest_first, self.eco.prj_duration_yrs, self.params.ls)

        capex = np.zeros(self.eco.prj_duration_yrs + 1, dtype=float)
        capex[invest_periods] = 1

        if self.params.residual_at_ls > 0:
            residual_periods = invest_periods[1:] if self.params.consider_preexisting else invest_periods
            capex[residual_periods] -= self.params.residual_at_ls

        capex[-1] = -1 * calc_residual_value(
            lifetime_remaining_frac=calc_lifetime_remaining(
                project_duration=self.eco.prj_duration_yrs,
                ls=self.params.ls,
                init_age=invest_first % self.params.ls,
            )
            / float(self.params.ls),
            depreciation=DEPRECIATION.LINEAR,
            residual_at_ls=self.params.residual_at_ls,
        )

        # apply capex cost change ratio
        capex *= self.params.ccr ** np.arange(self.eco.prj_duration_yrs + 1)

        return capex

    def _calc_cashflow_factor_preexisting(self) -> npt.NDArray:
        invest_first = 0 if self.params.consider_preexisting else self.params.ls - self.params.age_preexisting
        return self._calc_cashflow_factors(invest_first)

    def _calc_cashflow_factor_expansion(self) -> npt.NDArray:
        invest_first = 0
        return self._calc_cashflow_factors(invest_first)

    def _calc_cashflow_preexisting(self, size_preexisting: float | None) -> npt.NDArray:
        cost_size = size_preexisting * self.params.spec if size_preexisting else 0.0
        return self._calc_cashflow_factor_preexisting() * (cost_size + self.params.fix)

    def _calc_cashflow_expansion(self, size_expansion: float | None) -> npt.NDArray:
        cost_size = size_expansion * self.params.spec if size_expansion else 0.0
        return self._calc_cashflow_factor_expansion() * cost_size

    def _calc_cashflow(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> npt.NDArray:
        return self._calc_cashflow_preexisting(size_preexisting) + self._calc_cashflow_expansion(size_expansion)

    def evaluate(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> None:
        self._preexisting = self._calc_preexisting(size_preexisting=size_preexisting)
        self._expansion = self._calc_expansion(size_expansion=size_expansion)
        self._init = self._calc_init()
        super().evaluate(size_preexisting=size_preexisting, size_expansion=size_expansion, **kwargs)

    def _calc_spec_ep(self, **kwargs) -> float:
        return np.dot(
            self._calc_cashflow_factor_expansion() * self.params.spec,
            self.eco.discount_factors(self._OCCURS_AT),
        ) * self.eco.annuity_factor_apriori(self._OCCURS_AT)

    def get_preexisting(self, size_preexisting: float | None) -> float:
        return self._calc_preexisting(size_preexisting=size_preexisting)


class MntexEvaluator(CostEvaluator, CalculableYearlyElement, MntexElement):
    _OCCURS_AT = OccursAt.BEGIN

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: MntexParams,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
            **kwargs,
        )

    def _calc_yrl(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> float:
        size_preexisting = size_preexisting if size_preexisting else 0.0
        size_expansion = size_expansion if size_expansion else 0.0
        return self.params.spec * (size_preexisting + size_expansion) + self.params.fix

    def evaluate(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> None:
        super().evaluate(size_preexisting=size_preexisting, size_expansion=size_expansion, **kwargs)

    def _calc_spec_ep(self, **kwargs) -> float:
        return np.dot(
            np.full(self.eco.prj_duration_yrs + 1, self.params.spec),
            self.eco.discount_factors(self._OCCURS_AT),
        ) * self.eco.annuity_factor_apriori(self._OCCURS_AT)


class OpexEvaluator(PowerBasedEvaluator, OpexElement):
    _OCCURS_AT = OccursAt.END

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: OpexParams,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
            **kwargs,
        )


class CrevEvaluator(PowerBasedEvaluator, CrevElement):
    _OCCURS_AT = OccursAt.END

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: CrevParams,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
            **kwargs,
        )


@dataclass
class Evaluator(BlockElement):
    """
    EvaluatorBlock is a container for all Evaluators of a single component.
    """

    name: str

    eco: EcoParams

    name_size: str
    name_flow: str

    capex: CapexEvaluator
    mntex: MntexEvaluator
    opex: OpexEvaluator
    crev: CrevEvaluator

    CAPEX_EVALUATOR: ClassVar[Type[CapexEvaluator]] = CapexEvaluator
    MNTEX_EVALUATOR: ClassVar[Type[MntexEvaluator]] = MntexEvaluator
    OPEX_EVALUATOR: ClassVar[Type[OpexEvaluator]] = OpexEvaluator
    CREV_EVALUATOR: ClassVar[Type[CrevEvaluator]] = CrevEvaluator

    @classmethod
    def create(
        cls,
        name: str,
        eco: EcoParams,
        name_size: str,
        name_flow: str,
        params_capex: CapexParams,
        params_mntex: MntexParams,
        params_opex: OpexParams,
        params_crev: CrevParams,
    ) -> Self:
        return cls(
            name=name,
            eco=eco,
            name_size=name_size,
            name_flow=name_flow,
            capex=cls.CAPEX_EVALUATOR(name, eco, params_capex),
            mntex=cls.MNTEX_EVALUATOR(name, eco, params_mntex),
            opex=cls.OPEX_EVALUATOR(name, eco, params_opex),
            crev=cls.CREV_EVALUATOR(name, eco, params_crev),
        )

    @classmethod
    def _build_capex_params(
        cls,
        eco: EcoParams,
        spec: float,
        fix: float,
        ccr: float,
        consider_preexisting: bool,
        ls: int | None,
        age_preexisting: int,
        residual_at_ls: float,
    ) -> CapexParams:
        return CapexParams.create_from_plain(
            spec=spec,
            fix=fix,
            consider_preexisting=consider_preexisting,
            ls=eco.prj_duration_yrs if ls is None else ls,
            age_preexisting=age_preexisting,
            ccr=ccr,
            residual_at_ls=residual_at_ls,
        )

    @classmethod
    def _build_mntex_params(
        cls,
        spec: float,
        fix: float,
    ) -> MntexParams:
        return MntexParams.create_from_plain(spec=spec, fix=fix)

    @classmethod
    def _build_opex_params(
        cls,
        eco: EcoParams,
        data_dir: Path,
        spec: float | Path,
        fix: float,
        **kwargs,
    ) -> OpexParams:
        return OpexParams.create_from_plain(
            spec=spec,
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
    ) -> CrevParams:
        return CrevParams.create_from_plain(
            spec=spec,
            fix=fix,
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
        opex_fix: float = 0.0,
        # crev
        crev_spec: float | Path = 0.0,
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
            fix=opex_fix,
        )

        params_crev = cls._build_crev_params(
            eco=eco,
            data_dir=data_dir,
            spec=crev_spec,
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

    @property
    def spec_ep_invest(self):
        """
        Equivalent present specific costs for investments (cost per size)
        """
        return self.capex.spec_ep + self.mntex.spec_ep

    @property
    def spec_ep_operation(self):
        """
        Equivalent present specific costs for operation (cost per energy)
        """
        return self.opex.spec_ep + self.crev.spec_ep

    def evaluate(
        self,
        size_preexisting: float | None = None,
        size_expansion: float | None = None,
        flow: pd.Series | None = None,
        **kwargs,
    ) -> None:
        for attr in (self.capex, self.mntex, self.opex, self.crev):
            if attr:
                attr.evaluate(
                    size_preexisting=size_preexisting,
                    size_expansion=size_expansion,
                    flow=flow,
                    **kwargs,
                )

    def get_invest_preexisting(self, size_preexisting: float | None) -> float:
        return self.capex.get_preexisting(size_preexisting=size_preexisting)
