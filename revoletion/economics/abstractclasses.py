from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .params import EcoParams
from .utils import CostTypeDefinition, OccursAt


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

    def __init_subclass__(cls, **kwargs):
        # ensure that concrete subclasses define the _TYPE class attribute to specify their cost type
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "__abstractmethods__") and ABC not in cls.__bases__:
            if not any("_TYPE" in B.__dict__ for B in cls.mro()):
                raise TypeError(f"Concrete class {cls.__name__} must define '_TYPE'")

    def __init__(self, name: str, **kwargs):
        self.name = name

        self._cashflow: np.ndarray | None = None
        self._cashflow_dis: np.ndarray | None = None

        self._prj: float | None = None
        self._dis: float | None = None
        self._ann: float | None = None

    def _require_calculated(self, attr_name) -> Any:
        # helper method to ensure that results have been calculated before accessing them via the properties
        value = getattr(self, attr_name)
        if value is None:
            raise ValueError(f'Results need to be calculated before "{attr_name.strip("_")}" can be accessed.')
        return value

    @property
    def cashflow(self):
        return self._require_calculated("_cashflow")

    @property
    def cashflow_dis(self):
        return self._require_calculated("_cashflow_dis")

    @property
    def prj(self):
        return self._require_calculated("_prj")

    @property
    def dis(self):
        return self._require_calculated("_dis")

    @property
    def ann(self):
        return self._require_calculated("_ann")

    @property
    def _result_summary_prefix(self) -> str:
        """
        Define the prefix for the result summary
        """
        return f"{self._TYPE.value.label}_"

    @property
    def result_summary(self) -> pd.Series:
        """
        Return a pandas Series containing the scalar cost or revenue results of the element
        """
        return pd.Series(
            data={
                f"{self._result_summary_prefix}prj": self.prj,
                f"{self._result_summary_prefix}dis": self.dis,
                f"{self._result_summary_prefix}ann": self.ann,
            },
        )


class YearlyElement(BaseElement, ABC):
    """
    Base class for all elements with yearly occurring costs or revenues (Mntex, Opex, Crev).
    Add yearly costs or revenues (yrl) as an additional result and extend the result summary by this value.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)

        self._yrl: float | None = None

    @property
    def yrl(self) -> float:
        return self._require_calculated("_yrl")

    @property
    def result_summary(self) -> pd.Series:
        """
        Extend the result summary of the BaseElement by the yearly costs or revenues.
        """
        return pd.concat(
            [
                super().result_summary,
                pd.Series(
                    data={
                        f"{self._result_summary_prefix}yrl": self.yrl,
                    },
                ),
            ],
            axis=0,
        )


class TimeseriesElement(YearlyElement, ABC):
    """
    Base class for elements whose costs or revenues scale proportionally from the evaluation period to a one-year basis (Opex, Crev).
    Add evaluation period costs or revenues (eval) as an additional result and extend the result summary by this value.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)

        self._eval: float | None = None

    @property
    def eval(self) -> float:
        return self._require_calculated("_eval")

    @property
    def result_summary(self) -> pd.Series:
        """
        Extend the result summary of the YearlyElement by the evaluation period costs or revenues.
        """
        return pd.concat(
            [
                super().result_summary,
                pd.Series(
                    data={
                        f"{self._result_summary_prefix}eval": self.eval,
                    },
                ),
            ],
            axis=0,
        )


class CapexElement(BaseElement, ABC):
    """
    Base class for all capex elements.
    Defines the _TYPE class attribute.
    Add preexisting, expansion and init costs as additional results and extend the result summary by these values.
    """

    _TYPE = CostType.CAPEX

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)

        self._preexisting: float | None = None
        self._expansion: float | None = None
        self._init: float | None = None

    @property
    def preexisting(self) -> float:
        return self._require_calculated("_preexisting")

    @property
    def expansion(self) -> float:
        return self._require_calculated("_expansion")

    @property
    def init(self) -> float:
        return self._require_calculated("_init")

    @property
    def result_summary(self) -> pd.Series:
        return pd.concat(
            [
                super().result_summary,
                pd.Series(
                    data={
                        f"{self._result_summary_prefix}preexisting": self.preexisting,
                        f"{self._result_summary_prefix}expansion": self.expansion,
                        f"{self._result_summary_prefix}init": self.init,
                    },
                ),
            ],
            axis=0,
        )


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


class CalculableBaseElement(BaseElement, ABC):
    """
    Base class for all economic elements that calculate their own results instead of just aggregate them.
    """

    def __init__(self, name: str, eco: EcoParams, **kwargs):
        super().__init__(name, **kwargs)
        self.eco = eco

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

    def evaluate(self, *args, **kwargs):
        self._cashflow = self._calc_cashflow(*args, **kwargs)
        self._cashflow_dis = self._calc_cashflow_dis()
        self._prj = self._calc_prj()
        self._dis = self._calc_dis()
        self._ann = self._calc_ann()


class CalculableYearlyElement(CalculableBaseElement, YearlyElement, ABC):
    """
    Base class for all economic elements that calculate their own results and have yearly occurring costs or revenues (Mntex, Opex, Crev).
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
        self._yrl = self._calc_yrl(*args, **kwargs)
        super().evaluate(*args, **kwargs)


class CalculableTimeseriesElement(CalculableYearlyElement, TimeseriesElement, ABC):
    """
    Base class for all economic elements that calculate their own results, have yearly occurring costs or revenues and scale proportionally from the evaluation period to a one-year basis (Opex, Crev).
    """

    def __init__(self, name: str, eco: EcoParams, **kwargs):
        super().__init__(name=name, eco=eco, **kwargs)

    @abstractmethod
    def _calc_eval(self, *args, **kwargs) -> float: ...

    def _calc_yrl(self, *args, **kwargs) -> float:
        return self.eval / self.eco.eval_yr_rat

    def evaluate(self, *args, **kwargs):
        # calculate costs or revenues in evaluation timeframe -> used to calculate yearly costs
        self._eval = self._calc_eval(*args, **kwargs)
        super().evaluate(*args, **kwargs)


class BlockElement(ABC):
    """
    Base class for all block elements that aggregate multiple economic elements.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name

        self.capex: CapexElement | None = None
        self.mntex: YearlyElement | None = None
        self.opex: YearlyElement | None = None
        self.crev: YearlyElement | None = None

    @property
    def result_summary(self) -> pd.Series:
        return pd.concat(
            [
                self.capex.result_summary if self.capex is not None else pd.Series(),
                self.mntex.result_summary if self.mntex is not None else pd.Series(),
                self.opex.result_summary if self.opex is not None else pd.Series(),
                self.crev.result_summary if self.crev is not None else pd.Series(),
            ],
            axis=0,
        )
