from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from .params import (
    EcoParams,
    CostParams,
    CapexParams,
    MntexParams,
    TimeseriesParams,
    OpexParams,
    CrevParams,
) 
from .utils import CostTypeDefinition, OccursAt, Depreciation, transform_scalar_var, calc_lifetime_remaining, calc_residual_value, dot_scalar


class CostType(Enum):
    """
    Definition of all cost types.
    Each cost type is associated with a CostTypeDefinition that defines its label and the occurrence of the costs occur within a period (begin, mid or end).
    """

    CAPEX = CostTypeDefinition("capex", OccursAt.BEGIN)
    MNTEX = CostTypeDefinition("mntex", OccursAt.BEGIN)
    OPEX = CostTypeDefinition("opex", OccursAt.END)
    CREV = CostTypeDefinition("crev", OccursAt.END)
    TOTEX = CostTypeDefinition("totex", None)
    VALUE = CostTypeDefinition("value", None)
    ENERGY = CostTypeDefinition("energy", OccursAt.END)


class BaseElement(ABC):
    """
    Base class for all economic elements.
    It defines all common attributes and methods for the different cost types:
    cashflow: npt.NDArray - nominal cashflow per year without discounting
    cashflow_dis: npt.NDArray - discounted cashflow per year
    prj: float - total costs over the project duration (sum of cashflow)
    dis: float - total discounted costs over the project duration (sum of cashflow_dis)
    ann: float - annuity of the discounted costs over the project duration
    """

    _TYPE: CostType  # concrete subclasses need to define this class attribute to specify their cost type
    _AGGR_ATTRS = ["cashflow", "cashflow_dis", "prj", "dis", "ann"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # ensure that concrete subclasses define the _TYPE class attribute to specify their cost type
        if not hasattr(cls, "__abstractmethods__") and ABC not in cls.__bases__:
            if not any("_TYPE" in B.__dict__ for B in cls.mro()):
                raise TypeError(f"Concrete class {cls.__name__} must define '_TYPE'")
            
        # combine _AGGR_ATTRS of sublcass with parent class
        # use cls.__dict__.get("_AGGR_ATTRS", []) to avoid duplicating parent's list if current class does not define its own list
        cls._AGGR_ATTRS = cls.__dict__.get("_AGGR_ATTRS", []) + cls.__mro__[1]._AGGR_ATTRS

    def __init__(self, name: str, eco: EcoParams, **kwargs):
        self.name = name

        self.eco = eco

        self.cashflow: np.ndarray | None = None
        self.cashflow_dis: np.ndarray | None = None

        self.prj: float | None = None
        self.dis: float | None = None
        self.ann: float | None = None

    def _initialize_cumulative_attributes(self):
        self.cashflow = np.zeros(self.eco.prj_duration_yrs + 1)
        self.cashflow_dis = np.zeros(self.eco.prj_duration_yrs + 1)
        self.prj = 0.0
        self.dis = 0.0
        self.ann = 0.0

    def _aggregate_attribute(self, attribute_name: str, target: BaseElement) -> None:
        target_value = getattr(target, attribute_name)
        source_value = getattr(self, attribute_name)
        setattr(target, attribute_name, target_value + source_value)
    
    def aggregate(self, target: Node | None = None):
        target_aggregator = getattr(target, self._TYPE.value.label)

        for aggr_attr in self._AGGR_ATTRS:
            self._aggregate_attribute(attribute_name=aggr_attr, target=target_aggregator)

    def get_cashflow(self, discounted: bool = False) -> npt.NDArray:
        return self.cashflow_dis if discounted else self.cashflow

    @property
    def _result_summary_prefix(self) -> str:
        """
        Define the prefix for the result summary
        """
        return f"{self._TYPE.value.label}_"

    @property
    def result_summary(self) -> dict:
        """
        Return a dict containing the scalar cost or revenue results of the element
        """
        return {
            f"{self._result_summary_prefix}prj": self.prj,
            f"{self._result_summary_prefix}dis": self.dis,
            f"{self._result_summary_prefix}ann": self.ann,
        }


class YearlyElement(BaseElement, ABC):
    """
    Base class for all elements with yearly occurring costs or revenues (Mntex, Opex, Crev).
    Add yearly costs or revenues (yrl) as an additional result and extend the result summary by this value.
    """

    _AGGR_ATTRS = ["yrl"]

    def __init__(self, name: str, eco: EcoParams, **kwargs):
        super().__init__(name=name, eco=eco, **kwargs)

        self.yrl: float | None = None

    def _initialize_cumulative_attributes(self):
        self.yrl = 0.0
        super()._initialize_cumulative_attributes()

    @property
    def result_summary(self) -> dict:
        """
        Extend the result summary of the BaseElement by the yearly costs or revenues.
        """
        return {
            f"{self._result_summary_prefix}yrl": self.yrl,
            **super().result_summary,
        }


class TimeseriesElement(YearlyElement, ABC):
    """
    Base class for elements whose costs or revenues scale proportionally from the evaluation period to a one-year basis (Opex, Crev).
    Add evaluation period costs or revenues (eval) as an additional result and extend the result summary by this value.
    """

    _AGGR_ATTRS = ["eval"]

    def __init__(self, name: str, eco: EcoParams, **kwargs):
        super().__init__(name=name, eco=eco, **kwargs)

        self.eval: float | None = None

    def _initialize_cumulative_attributes(self):
        self.eval = 0.0
        super()._initialize_cumulative_attributes()

    @property
    def result_summary(self) -> dict:
        """
        Extend the result summary of the YearlyElement by the evaluation period costs or revenues.
        """
        return {
            f"{self._result_summary_prefix}eval": self.eval,
            **super().result_summary,
        }


class CapexElement(BaseElement, ABC):
    """
    Base class for all capex elements.
    Defines the _TYPE class attribute.
    Add preexisting, expansion and init costs as additional results and extend the result summary by these values.
    """

    _TYPE = CostType.CAPEX
    _AGGR_ATTRS = ["preexisting", "expansion", "init"]

    def __init__(self, name: str, eco: EcoParams, **kwargs):
        super().__init__(name=name, eco=eco, **kwargs)

        self.preexisting: float | None = None
        self.expansion: float | None = None
        self.init: float | None = None

    def _initialize_cumulative_attributes(self):
        self.preexisting = 0.0
        self.expansion = 0.0
        self.init = 0.0
        super()._initialize_cumulative_attributes()

    @property
    def result_summary(self) -> dict:
        return {
            f"{self._result_summary_prefix}preexisting": self.preexisting,
            f"{self._result_summary_prefix}expansion": self.expansion,
            f"{self._result_summary_prefix}init": self.init,
            **super().result_summary,
        }


class MntexElement(YearlyElement, ABC):
    """
    Base class for all mntex elements.
    Defines the _TYPE class attribute.
    """

    _TYPE = CostType.MNTEX


class OpexElement(TimeseriesElement, ABC):
    """
    Base class for all opex elements.
    Defines the _TYPE class attribute.
    """

    _TYPE = CostType.OPEX


class CrevElement(TimeseriesElement, ABC):
    """
    Base class for all crev elements.
    Defines the _TYPE class attribute.
    """

    _TYPE = CostType.CREV


class TotexElement(BaseElement, ABC):
    """
    Base class for all totex elements.
    Defines the _TYPE class attribute.
    """

    _TYPE = CostType.TOTEX


class ValueElement(BaseElement, ABC):
    """
    Base class for all value elements.
    Defines the _TYPE class attribute.
    """

    _TYPE = CostType.VALUE


class BaseEvaluator(BaseElement, ABC):
    """
    Base class for all economic elements that evaluate their own results instead of just aggregate them from other nodes.
    Depending on the functionality, there are two different Evaluator classes:
        PrimaryEvaluator for Capex, Mntex, Opex, Crev -> evaluate results stand-alone
        SecondaryEvaluator for Totex, Value -> evaluate results based on other Evaluators of the same Evaluator
    """

    def __init__(self, name: str, eco: EcoParams, **kwargs):
        super().__init__(name, eco=eco, **kwargs)

    @abstractmethod
    def _calc_cashflow(self, *args, **kwargs) -> npt.NDArray: ...

    @abstractmethod
    def _calc_cashflow_dis(self) -> npt.NDArray: ...

    @abstractmethod
    def _calc_prj(self) -> float: ...

    @abstractmethod
    def _calc_dis(self) -> float: ...

    @abstractmethod
    def _calc_ann(self) -> float: ...

    def evaluate(self, *args, **kwargs):
        self.cashflow = self._calc_cashflow(*args, **kwargs)
        self.cashflow_dis = self._calc_cashflow_dis()
        self.prj = self._calc_prj()
        self.dis = self._calc_dis()
        self.ann = self._calc_ann()


class PrimaryEvaluator(BaseEvaluator, ABC):
    def __init__(self, name: str, eco: EcoParams, **kwargs):
        super().__init__(name=name, eco=eco, **kwargs)

    @abstractmethod
    def _calc_cashflow(self, *args, **kwargs) -> npt.NDArray: ...

    def _calc_cashflow_dis(self) -> npt.NDArray:
        return self.cashflow * self.eco.discount_factors(self._TYPE.value.occurs_at)

    def _calc_prj(self) -> float:
        return np.sum(self.cashflow)

    def _calc_dis(self) -> float:
        return np.sum(self.cashflow_dis)

    def _calc_ann(self) -> float:
        return self.dis * self.eco.annuity_factor(self._TYPE.value.occurs_at)
    
    @classmethod
    @abstractmethod
    def _build_kwargs_from_params(cls, params: CostParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict: ...

    @classmethod
    def create_from_params(cls, name: str, eco: EcoParams, params: CostParams, data_dir: Path, **kwargs) -> Self:
        # use _build_kwargs_from_params to easily add parameters for specific evaluators without changing this method
        kwargs = cls._build_kwargs_from_params(params=params, eco=eco, data_dir=data_dir, **kwargs)
        return cls(name=name, eco=eco, **kwargs)

    def _calc_ep_factor(self, cashflow_factors: npt.NDArray, **kwargs) -> float:
        # calculate the factor which is multiplied with the specific costs to get the equivalent present specific costs used by the optimization problem.
        # this factor scales all specific costs to their net present costs of the whole project duration.
        return (
            np.dot(cashflow_factors, self.eco.discount_factors(self._TYPE.value.occurs_at))
            if self.eco.compensate_sim_prj
            else 1.0
        )
    
    @abstractmethod
    def _calc_spec_ep(self, **kwargs) -> float | pd.Series: ...

    @property
    def spec_ep(self) -> float | pd.Series:
        # return the equivalent present specific costs for optimization.
        # for capex and mntex: component size specific costs
        # for opex and crev: power flow specific costs.
        return self._calc_spec_ep()
    

class YearlyEvaluator(PrimaryEvaluator, YearlyElement, ABC):
    """
    Base class for all economic elements that evaluate their own results and have yearly occurring costs or revenues (Mntex, Opex, Crev).
    """

    def __init__(self, name: str, eco: EcoParams, **kwargs):
        super().__init__(name=name, eco=eco, **kwargs)

        # neglect last year as it is just for residual value of capex
        self._cashflow_factors = np.full(self.eco.prj_duration_yrs + 1, 1.0)
        self._cashflow_factors[-1] = 0.0

    @property
    def cashflow_factors(self) -> npt.NDArray:
        # access element using property to prevent any modifications to the cashflow factors after initialization
        return self._cashflow_factors

    @abstractmethod
    def _calc_yrl(self, *args, **kwargs) -> float: ...

    def _calc_cashflow(self, *args, **kwargs) -> npt.NDArray:
        return self.cashflow_factors * self.yrl

    def evaluate(self, *args, **kwargs):
        # calculate yearly costs or revenues -> used to calculate cashflow
        self.yrl = self._calc_yrl(*args, **kwargs)
        super().evaluate(*args, **kwargs)


class TimeseriesEvaluator(YearlyEvaluator, TimeseriesElement, ABC):
    """
    Base class for all economic elements that evaluate their own results, have yearly occurring costs or revenues and scale proportionally from the evaluation period to a one-year basis (Opex, Crev).
    """

    def __init__(
        self,
        name: str,
        eco: EcoParams,
        spec_energy: pd.Series,
        spec_dist: pd.Series,
        spec_time: pd.Series,
        fix: float,
        **kwargs
    ):
        super().__init__(name=name, eco=eco, **kwargs)
        
        self.spec_energy = spec_energy
        self.spec_dist = spec_dist
        self.spec_time = spec_time
        self.fix = fix

    @classmethod
    def _build_kwargs_from_params(cls, params: TimeseriesParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict:
        return dict(
            spec_energy=transform_scalar_var(value=params.spec_energy, dti=eco.dti_sim, data_dir=data_dir),
            spec_dist=transform_scalar_var(value=params.spec_dist, dti=eco.dti_sim, data_dir=data_dir),
            spec_time=transform_scalar_var(value=params.spec_time, dti=eco.dti_sim, data_dir=data_dir),
            fix=params.fix,
            **kwargs,
        )

    def _calc_eval(self, power: pd.Series | None, dist: pd.Series | None, time: pd.Series | None, **kwargs) -> float:
        # evaluate costs related to the provided power flow
        cost_energy = (
            dot_scalar(
                self.spec_energy.to_numpy(),
                power[self.eco.dti_eval].to_numpy(),
            ) * self.eco.timestep_hours
            if power is not None
            else 0.0
        )

        # evaluate costs related to the provided distances driven
        cost_dist = (
            dot_scalar(
                self.spec_dist.to_numpy(),
                dist[self.eco.dti_eval].to_numpy(),
            )
            if dist is not None
            else 0.0
        )

        # evaluate costs related to the provided time of use
        cost_time = (
            dot_scalar(
                self.spec_time.to_numpy(),
                time[self.eco.dti_eval].to_numpy(),
            ) * self.eco.timestep_hours
            if time is not None
            else 0.0
        )

        return cost_energy + cost_dist + cost_time

    def _calc_yrl(self, *args, **kwargs) -> float:
        # fix opex/crev occur on a yearly basis
        return self.fix + self.eval / self.eco.eval_yr_rat

    def evaluate(self, power: pd.Series | None, dist: pd.Series | None = None, time: pd.Series | None = None, *args, **kwargs):
        # evaluate costs or revenues in evaluation timeframe -> used to evaluate yearly costs
        self.eval = self._calc_eval(power=power, dist=dist, time=time, *args, **kwargs)
        super().evaluate(*args, **kwargs)

    def _calc_spec_ep(self, **kwargs) -> pd.Series:
        # include linear scaling from simulation duration to year in cashflow_factors for correct spec_ep calculation
        factor_ep = self._calc_ep_factor(cashflow_factors=self.cashflow_factors / self.eco.sim_yr_rat)
        return self.spec_energy * factor_ep
    

class CapexEvaluator(PrimaryEvaluator, CapexElement):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        spec: float,
        fix: float,
        consider_preexisting: bool,
        ls: int,
        age_preexisting: int,
        ccr: float,
        residual_at_ls: float,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            **kwargs,
        )

        self.spec = spec
        self.fix = fix
        self.consider_preexisting = consider_preexisting
        self.ls = ls
        self.age_preexisting = age_preexisting
        self.ccr = ccr
        self.residual_at_ls = residual_at_ls

        if self.consider_preexisting and self.age_preexisting != 0:
            raise ValueError(f"If consider_preexisting is True, age_preexisting must be 0, got {self.age_preexisting}")
        if self.age_preexisting >= self.ls:
            raise ValueError(
                f"age_preexisting must be smaller than ls, got age_preexisting={self.age_preexisting} and ls={self.ls}"
            )

    @classmethod
    def _build_kwargs_from_params(cls, params: CapexParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict:
        return dict(
            spec=params.spec,
            fix=params.fix,
            consider_preexisting=params.consider_preexisting,
            ls=params.ls if params.ls is not None else eco.prj_duration_yrs,
            age_preexisting=params.age_preexisting,
            ccr=params.ccr,
            residual_at_ls=params.residual_at_ls,
            **kwargs,
        )

    def _calc_preexisting(self, size_preexisting: float | None) -> float:
        cost_size = size_preexisting * self.spec if size_preexisting else 0.0
        return (cost_size + self.fix) * int(self.consider_preexisting)

    def _calc_expansion(self, size_expansion: float | None) -> float:
        return size_expansion * self.spec if size_expansion else 0.0

    def _calc_init(self) -> float:
        return self.preexisting + self.expansion

    def _calc_cashflow_factors(self, invest_first: int, **kwargs) -> npt.NDArray:
        invest_periods = np.arange(invest_first, self.eco.prj_duration_yrs, self.ls)

        capex = np.zeros(self.eco.prj_duration_yrs + 1, dtype=float)
        capex[invest_periods] = 1

        if self.residual_at_ls > 0:
            residual_periods = invest_periods[1:] if self.consider_preexisting else invest_periods
            capex[residual_periods] -= self.residual_at_ls

        capex[-1] = -1 * calc_residual_value(
            lifetime_remaining_frac=calc_lifetime_remaining(
                project_duration=self.eco.prj_duration_yrs,
                ls=self.ls,
                init_age=invest_first % self.ls,
            )
            / float(self.ls),
            depreciation=Depreciation.LINEAR,
            residual_at_ls=self.residual_at_ls,
        )

        # apply capex cost change ratio
        capex *= self.ccr ** np.arange(self.eco.prj_duration_yrs + 1)

        return capex

    def _calc_cashflow_factor_preexisting(self) -> npt.NDArray:
        invest_first = 0 if self.consider_preexisting else self.ls - self.age_preexisting
        return self._calc_cashflow_factors(invest_first)

    def _calc_cashflow_factor_expansion(self) -> npt.NDArray:
        invest_first = 0
        return self._calc_cashflow_factors(invest_first)

    def _calc_cashflow_preexisting(self, size_preexisting: float | None) -> npt.NDArray:
        cost_size = size_preexisting * self.spec if size_preexisting else 0.0
        return self._calc_cashflow_factor_preexisting() * (cost_size + self.fix)

    def _calc_cashflow_expansion(self, size_expansion: float | None) -> npt.NDArray:
        cost_size = size_expansion * self.spec if size_expansion else 0.0
        return self._calc_cashflow_factor_expansion() * cost_size

    def _calc_cashflow(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> npt.NDArray:
        return self._calc_cashflow_preexisting(size_preexisting) + self._calc_cashflow_expansion(size_expansion)

    def evaluate(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> None:
        self.preexisting = self._calc_preexisting(size_preexisting=size_preexisting)
        self.expansion = self._calc_expansion(size_expansion=size_expansion)
        self.init = self._calc_init()
        super().evaluate(size_preexisting=size_preexisting, size_expansion=size_expansion, **kwargs)

    def _calc_spec_ep(self, **kwargs) -> float:
        factor_ep = self._calc_ep_factor(cashflow_factors=self._calc_cashflow_factor_expansion())

        return self.spec * factor_ep

    def get_preexisting(self, size_preexisting: float | None) -> float:
        return self._calc_preexisting(size_preexisting=size_preexisting)
    

class MntexEvaluator(YearlyEvaluator, MntexElement):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        spec: float,
        fix: float,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            **kwargs,
        )

        self.spec = spec
        self.fix = fix

    @classmethod
    def _build_kwargs_from_params(cls, params: MntexParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict:
        return dict(spec=params.spec, fix=params.fix, **kwargs)

    def _calc_yrl(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> float:
        return self.spec * ((size_preexisting or 0.0) + (size_expansion or 0.0)) + self.fix

    def evaluate(self, size_preexisting: float | None, size_expansion: float | None, **kwargs) -> None:
        super().evaluate(size_preexisting=size_preexisting, size_expansion=size_expansion, **kwargs)

    def _calc_spec_ep(self, **kwargs) -> float:
        factor_ep = self._calc_ep_factor(cashflow_factors=self.cashflow_factors)

        return self.spec * factor_ep


class OpexEvaluator(TimeseriesEvaluator, OpexElement):
    def __init__(
        self,
        name: str,
        eco: EcoParams,
        spec_energy: pd.Series,
        spec_dist: pd.Series,
        spec_time: pd.Series,
        fix: float,
        spec_peak: float,
        frac_peak: float,
        **kwargs,
    ):
        super().__init__(
            name=name,
            eco=eco,
            spec_energy=spec_energy,
            spec_dist=spec_dist,
            spec_time=spec_time,
            fix=fix,
            **kwargs,
        )

        self.spec_peak = spec_peak
        self.frac_peak = frac_peak

    @classmethod
    def _build_kwargs_from_params(cls, params: OpexParams, eco: EcoParams, data_dir: Path, **kwargs) -> dict:
        return dict(
            **super()._build_kwargs_from_params(params=params, eco=eco, data_dir=data_dir, **kwargs),
            spec_peak=params.spec_peak,
            frac_peak=params.frac_peak,
        )

    def _calc_spec_ep_peak(self, **kwargs) -> float:
        factor_ep = self._calc_ep_factor(cashflow_factors=self.cashflow_factors / self.eco.sim_yr_rat)
        return self.spec_peak * self.frac_peak * factor_ep

    @property
    def spec_ep_peak(self) -> float:
        return self._calc_spec_ep_peak()

    def _calc_eval(
        self,
        power: pd.Series | None,
        dist: pd.Series | None,
        time: pd.Series | None,
        power_peak: float | None = None,
        **kwargs,
    ) -> float:
        return (
            super()._calc_eval(power=power, dist=dist, time=time, **kwargs) + 
            (power_peak * self.spec_peak * self.frac_peak if power_peak is not None else 0.0)
        )

    def evaluate(
        self,
        power: pd.Series | None,
        dist: pd.Series | None = None,
        time: pd.Series | None = None,
        power_peak: float | None = None,
        **kwargs,
    ) -> None:
        super().evaluate(power=power, dist=dist, time=time, power_peak=power_peak, **kwargs)


class CrevEvaluator(TimeseriesEvaluator, CrevElement):
    pass


class SecondaryEvaluator(BaseEvaluator, ABC):
    _COMPONENTS = {}  # Evaluator.key1 * value1 + Evaluator.key2 * value2 + ...

    def __init__(self, name: str, eco: EcoParams, evaluator: Evaluator | None = None, **kwargs):
        super().__init__(name=name, eco=eco, **kwargs)
        self.evaluator = evaluator

    def set_evaluator(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def _calc_attribute(self, attribute_name: str):
        return (
            # get starting value (required if all components are None, e.g. for totex if there are only revenues)
            getattr(self, attribute_name) +
            sum(
                factor * getattr(component_obj, attribute_name)
                for component, factor in self._COMPONENTS.items()
                if (component_obj := getattr(self.evaluator, component)) is not None
            )
        )

    def _calc_cashflow(self, *args, **kwargs) -> npt.NDArray:
        return self._calc_attribute(attribute_name="cashflow")

    def _calc_cashflow_dis(self) -> npt.NDArray:
        return self._calc_attribute(attribute_name="cashflow_dis")

    def _calc_prj(self) -> float:
        return self._calc_attribute(attribute_name="prj")

    def _calc_dis(self) -> float:
        return self._calc_attribute(attribute_name="dis")

    def _calc_ann(self) -> float:
        return self._calc_attribute(attribute_name="ann")
    
    def evaluate(self, *args, **kwargs):
        self._initialize_cumulative_attributes()
        super().evaluate(*args, **kwargs)


class TotexEvaluator(SecondaryEvaluator, TotexElement):
    # totex = capex + mntex + crev
    _COMPONENTS = {"capex": 1.0, "mntex": 1.0, "opex": 1.0}


class ValueEvaluator(SecondaryEvaluator, TotexElement):
    # value = crev - totex
    _COMPONENTS = {"crev": 1.0, "totex": -1.0}


class Node(ABC):
    """
    Base class for all block elements that aggregate multiple economic elements.
    """

    def __init__(
            self,
            name: str,
            capex: CapexElement,
            mntex: MntexElement,
            opex: OpexElement,
            crev: CrevElement,
            totex: TotexElement,
            value: ValueElement,
            **kwargs,
        ):
        self.name = name
        self.capex = capex
        self.mntex = mntex
        self.opex = opex
        self.crev = crev
        self.totex= totex
        self.value = value

    @classmethod
    @abstractmethod
    def create(cls, name: str, **kwargs) -> Self: ...

    def aggregate(self, target: Aggregator | None = None) -> None:
        # target can be None -> do not aggregate any further
        if not target:      
            return
        
        for component in [self.capex, self.mntex, self.opex, self.crev, self.totex, self.value]:
            if component:
                component.aggregate(target)

    @property
    def result_summary(self) -> dict:
        result_summary = {}

        for component in [self.capex, self.mntex, self.opex, self.crev, self.totex, self.value]:
            if component is not None:
                result_summary.update(component.result_summary)

        return result_summary


class Aggregator(Node):
    def __init__(
            self,
            name: str,
            capex: CapexElement,
            mntex: MntexElement,
            opex: OpexElement,
            crev: CrevElement,
            totex: TotexElement,
            value: ValueElement,
            **kwargs
        ):
        super().__init__(name=name, capex=capex, mntex=mntex, opex=opex, crev=crev, totex=totex, value=value, **kwargs)

        self.nodes = {}

    @classmethod
    def create(cls, name: str, eco: EcoParams, **kwargs) -> Self:
        return cls(
            name=name,
            capex=CapexElement(name="capex", eco=eco),
            mntex=MntexElement(name="mntex", eco=eco),
            opex=OpexElement(name="opex", eco=eco),
            crev=CrevElement(name="crev", eco=eco),
            totex=TotexElement(name="totex", eco=eco),
            value=ValueElement(name="value", eco=eco),
        )

    def add_node(self, node: Node) -> None:
        """
        Add a new node to the aggregator.
        """
        if node.name in self.nodes.keys():
            raise ValueError(f"node with name \"{node.name}\" is already in nodes of Aggregator \"{self.name}\".")
        self.nodes[node.name] = node

    def _reset_components(self):
        for component in [self.capex, self.mntex, self.opex, self.crev, self.totex, self.value]:
            component._initialize_cumulative_attributes()

    def aggregate(self, target: Aggregator | None = None) -> None:
        self._reset_components()
        for node in self.nodes.values():
            node.aggregate(target=self)
        super().aggregate(target=target)


class Evaluator(Node):
    def __init__(
            self,
            name: str,
            name_size: str | None,
            name_flow: str | None,
            capex: CapexEvaluator,
            mntex: MntexEvaluator,
            opex: OpexEvaluator,
            crev: CrevEvaluator,
            totex: TotexEvaluator,
            value: ValueEvaluator,
            **kwargs
        ):
        super().__init__(name=name, capex=capex, mntex=mntex, opex=opex, crev=crev, totex=totex, value=value, **kwargs)
        self.name_size = name_size
        self.name_flow = name_flow

        self.totex.set_evaluator(self)
        self.value.set_evaluator(self)

    @classmethod
    def create(
        cls,
        name: str,
        eco: EcoParams,
        data_dir: Path,
        name_size: str | None = None,
        name_flow: str | None = None,
        capex: CapexParams | None = None,
        mntex: MntexParams | None = None,
        opex: OpexParams | None = None,
        crev: CrevParams | None = None,
    ) -> Self:
        return cls(
            name=name,
            eco=eco,
            name_size=name_size,
            name_flow=name_flow,
            capex=CapexEvaluator.create_from_params(name, eco, capex, data_dir) if capex else None,
            mntex=MntexEvaluator.create_from_params(name, eco, mntex, data_dir) if mntex else None,
            opex=OpexEvaluator.create_from_params(name, eco, opex, data_dir) if opex else None,
            crev=CrevEvaluator.create_from_params(name, eco, crev, data_dir) if crev else None,
            totex=TotexEvaluator(name, eco),
            value=ValueEvaluator(name, eco),
        )

    def evaluate(
        self,
        size_preexisting: float | None = None,
        size_expansion: float | None = None,
        power: pd.Series | None = None,
        dist: pd.Series | None = None,
        time: pd.Series | None = None,
        power_peak: float | None = None,
        **kwargs,
    ) -> None:
        if time is not None:
            time = time.astype(bool).astype(int)  # ensure that time is a binary indicator (1 > 0, else 0)

        for attr in (self.capex, self.mntex, self.opex, self.crev, self.totex, self.value):
            if attr:
                attr.evaluate(
                    size_preexisting=size_preexisting,
                    size_expansion=size_expansion,
                    power=power,
                    dist=dist,
                    time=time,
                    power_peak=power_peak,
                    **kwargs,
                )
        
    def aggregate(self, target: Aggregator | None = None):
        super().aggregate(target=target)

    @property
    def spec_ep_invest(self):
        """
        Equivalent present specific costs for investments (cost per size)
        """
        return sum(component.spec_ep for component in (self.capex, self.mntex) if component is not None)

    @property
    def spec_ep_operation(self):
        """
        Equivalent present specific costs for operation (cost per energy)
        """
        return self.opex.spec_ep if self.opex is not None else 0.0

    @property
    def spec_ep_peak(self) -> float:
        """
        Equivalent present specific costs for peak power (cost per power)
        """
        return self.opex.spec_ep_peak if self.opex is not None else 0.0
    
    def get_invest_preexisting(self, size_preexisting: float | None) -> float:
        return self.capex.get_preexisting(size_preexisting=size_preexisting) if self.capex is not None else 0.0
