from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from .utils import annuity, OccursAt


class BaseEcoElement(ABC):
    """
    Base class for all economic elements.
    """

    def __init__(self, name: str):
        self.name = name

        self._cashflow: np.ndarray | None = None
        self._cashflow_dis: np.ndarray | None = None

        self._prj: float | None = None
        self._dis: float | None = None
        self._ann: float | None = None

    def _require_calculated(self, attr_name) -> Any:
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


class CalculableEcoElement(BaseEcoElement, ABC):
    _OCCURS_AT = OccursAt.BEGIN
    """
    Base class for all economic elements that calculate their own results instead of just aggregate them.
    """

    def __init__(self, name: str, eco: EcoParams):
        super().__init__(name)
        self.eco = eco

    @abstractmethod
    def _calc_cashflow(self, *args, **kwargs) -> npt.NDArray: ...

    def _calc_cashflow_dis(self) -> npt.NDArray:
        return self.cashflow * self.eco.discount_factors(self._OCCURS_AT)

    def _calc_prj(self) -> float:
        return np.sum(self.cashflow)

    def _calc_dis(self) -> float:
        return np.sum(self.cashflow_dis)

    def _calc_ann(self) -> float:
        return self.dis * self.eco.annuity_factor(self._OCCURS_AT)

    def evaluate(self, *args, **kwargs):
        self._cashflow = self._calc_cashflow_dis()
        self._prj = self._calc_prj()
        self._dis = self._calc_dis()
        self._ann = self._calc_ann()


class CapexElement(BaseEcoElement, ABC):
    """
    Base class for all capex elements.
    """

    def __init__(self, name: str):
        super().__init__(name=name)

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


class YearlyElement(BaseEcoElement, ABC):
    """
    Base class for all elements with yearly occurring costs or revenues (Mntex, Opex, Crev).
    """

    def __init__(self, name: str):
        super().__init__(name=name)

        self._yrl: float | None = None

    @property
    def yrl(self) -> float:
        return self._require_calculated("_yrl")


class PowerBasedElement(YearlyElement, ABC):
    """
    Base class for elements whose costs or revenues scale proportionally from the evaluation period to a one-year basis (Opex, Crev).
    """

    def __init__(self, name: str):
        super().__init__(name=name)

        self._eval: float | None = None

    @property
    def eval(self) -> float:
        return self._require_calculated("_eval")


class BlockElement(ABC):
    """
    Base class for all block elements.
    """

    def __init__(self, name: str):
        self.name = name

        self.capex: CapexElement | None = None
        self.mntex: YearlyElement | None = None
        self.opex: YearlyElement | None = None
        self.crev: YearlyElement | None = None
