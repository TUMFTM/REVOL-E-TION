from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

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
    ):
        super().__init__(name=name, eco=eco)
        self.eco = eco
        self.params = params

    @abstractmethod
    def _calc_spec_ep(self) -> float | pd.Series: ...

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
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )

    def _calc_eval(self, flow: pd.Series) -> float:
        return (
            np.dot(self.params.spec.to_numpy(), flow[self.eco.dti_eval].to_numpy()) * self.eco.timestep_hours
            + self.params.fix
        )

    def evaluate(self, flow: pd.Series, **kwargs):
        super().evaluate(flow=flow, **kwargs)

    def _calc_spec_ep(self) -> pd.Series:
        # calculate annuity due factor to compensate operation costs for difference between simulation and project time
        factor_operation_ep = (1 / self.eco.eval_yr_rat) if self.eco.compensate_sim_prj else 1

        return self.params.spec * factor_operation_ep


class CapexEvaluator(CostEvaluator, CapexElement):
    _OCCURS_AT = OccursAt.BEGIN

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: CapexParams,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )

    def _calc_cashflow_factors(self, invest_first: int) -> npt.NDArray:
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

    def _calc_cashflow_preexisting(self, size_preexisting: float) -> npt.NDArray:
        return self._calc_cashflow_factor_preexisting() * (self.params.spec * size_preexisting + self.params.fix)

    def _calc_cashflow_expansion(self, size_expansion: float) -> npt.NDArray:
        return self._calc_cashflow_factor_expansion() * (self.params.spec * size_expansion + self.params.fix)

    def _calc_cashflow(self, size_preexisting: float, size_expansion: float) -> npt.NDArray:
        return self._calc_cashflow_preexisting(size_preexisting) + self._calc_cashflow_expansion(size_expansion)

    def evaluate(self, size_preexisting: float, size_expansion: float, **kwargs) -> None:
        super().evaluate(size_preexisting=size_preexisting, size_expansion=size_expansion, **kwargs)

    def _calc_spec_ep(self) -> float:
        return np.dot(
            self._calc_cashflow_factor_expansion() * self.params.spec,
            self.eco.discount_factors(self._OCCURS_AT),
        ) * self.eco.annuity_factor_apriori(self._OCCURS_AT)

    # ToDo: fix these properties -> introduce methods to calculate preexisting and expansion
    @cached_property
    def preexisting(self) -> float:
        return 0.0
        return (self.results.size_preexisting * self.params.spec + self.params.fix) * int(
            self.params.consider_preexisting
        )

    @cached_property
    def expansion(self) -> float:
        return 0.0
        return self.results.size_expansion * self.params.spec + self.params.fix

    @cached_property
    def init(self) -> float:
        return self.preexisting + self.expansion


class MntexEvaluator(CostEvaluator, CalculableYearlyElement, MntexElement):
    _OCCURS_AT = OccursAt.BEGIN

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: MntexParams,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )

    def _calc_yrl(self, size_preexisting: float, size_expansion: float, *args, **kwargs) -> float:
        return self.params.spec * (size_preexisting + size_expansion) + self.params.fix

    def evaluate(self, size_preexisting: float, size_expansion: float, **kwargs) -> None:
        super().evaluate(size_preexisting=size_preexisting, size_expansion=size_expansion, **kwargs)

    def _calc_spec_ep(self) -> float:
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
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )


class CrevEvaluator(PowerBasedEvaluator, CrevElement):
    _OCCURS_AT = OccursAt.END

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        params: CrevParams,
    ):
        super().__init__(
            name=name,
            eco=eco,
            params=params,
        )


@dataclass(slots=True)
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

    @classmethod
    def create(
        cls,
        name: str,
        eco: EcoParams,
        data_dir: Path,
        capex_spec: float = 0.0,
        capex_fix: float = 0.0,
        capex_ccr: float = 1.0,
        consider_preexisting: bool = True,
        ls: int = None,
        age_preexisting: int = 0,
        capex_residual_at_ls: float = 0.0,
        mntex_spec: float = 0.0,
        mntex_fix: float = 0.0,
        opex_spec: float | Path = 0.0,
        opex_fix: float = 0.0,
        crev_spec: float | Path = 0.0,
        crev_fix: float = 0.0,
        size_name: str = None,
        flow_name: str = None,
    ) -> Self:
        params_capex = CapexParams(
            spec=capex_spec,
            fix=capex_fix,
            consider_preexisting=consider_preexisting,
            ls=eco.prj_duration_yrs if ls is None else ls,
            age_preexisting=age_preexisting,
            ccr=capex_ccr,
            residual_at_ls=capex_residual_at_ls,
        )

        params_mntex = MntexParams(
            spec=mntex_spec,
            fix=mntex_fix,
        )

        params_opex = OpexParams.create_from_plain(
            spec=opex_spec,
            fix=opex_fix,
            dti_sim=eco.dti_sim,
            data_dir=data_dir,
        )

        params_crev = CrevParams.create_from_plain(
            spec=crev_spec,
            fix=crev_fix,
            dti_sim=eco.dti_sim,
            data_dir=data_dir,
        )

        capex = CapexEvaluator(name=name, eco=eco, params=params_capex)
        mntex = MntexEvaluator(name=name, eco=eco, params=params_mntex)
        opex = OpexEvaluator(name=name, eco=eco, params=params_opex)
        crev = CrevEvaluator(name=name, eco=eco, params=params_crev)

        size_name = size_name
        flow_name = flow_name

        return cls(
            name=name,
            eco=eco,
            name_size=size_name,
            name_flow=flow_name,
            capex=capex,
            mntex=mntex,
            opex=opex,
            crev=crev,
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
        sizes: dict,
        flows: pd.DataFrame,
    ) -> None:
        if self.name_size:
            size_preexisting = sizes[self.name_size].preexisting
            size_expansion = sizes[self.name_size].expansion
        else:
            size_preexisting = 0.0
            size_expansion = 0.0

        if self.name_flow:
            flow = flows[self.name_flow]
        else:
            flow = pd.Series(0.0, index=self.eco.dti_eval, dtype=float)

        self.capex.evaluate(size_preexisting=size_preexisting, size_expansion=size_expansion)
        self.mntex.evaluate(size_preexisting=size_preexisting, size_expansion=size_expansion)
        self.opex.evaluate(flow=flow)
        self.crev.evaluate(flow=flow)
