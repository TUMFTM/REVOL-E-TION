from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class EcoElement(ABC):
    """
    Base class for all economic elements.

    This abstract base class defines the common interface and derived values
    that every economic element in the model must provide.

    Attributes
    ----------
    name : str
        Name of the economic element.

    cashflow : np.ndarray
        Cashflow of the economic element per period.

    cashflow_dis : np.ndarray
        Discounted cashflow per period.

    prj : float
        Project value (undiscounted sum of cashflows).

    dis : float
        Discounted project value.

    ann : float
        Annuity value of the economic element.
    """

    name: str

    @property
    @abstractmethod
    def cashflow(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def cashflow_dis(self) -> np.ndarray: ...

    @property
    def prj(self) -> float:
        return self.cashflow.sum()

    @property
    @abstractmethod
    def dis(self) -> float: ...

    @property
    @abstractmethod
    def ann(self) -> float: ...


@dataclass
class CapexElement(EcoElement, ABC):
    """
    Base class for all capex elements.

    This abstract base class defines the common interface and properties
    that every capex element in the model must provide in addition to the attributes defined in EcoElement.

    Attributes
    ----------
        name : str
        Name of the economic element.

    cashflow : np.ndarray
        Cashflow of the economic element per period.

    cashflow_dis : np.ndarray
        Discounted cashflow per period.

    prj : float
        Project value (undiscounted sum of cashflows).

    dis : float
        Discounted project value.

    ann : float
        Annuity value of the economic element.

    preexisting : float
        Capex of the preexisting installed component capacity.

    expansion : float
        Capex of the additional installed component capacity.

    init : float
        Capex of the initial installation (preexisting + expansion).
    """

    @property
    @abstractmethod
    def preexisting(self) -> float: ...

    @property
    @abstractmethod
    def expansion(self) -> float: ...

    @property
    @abstractmethod
    def init(self) -> float: ...


@dataclass
class YearlyElement(EcoElement, ABC):
    """
    Base class for all elements with yearly occurring costs or revenues (Mntex, Opex, Crev).

    This abstract base class defines the common interface and properties
    that every yearly element in the model must provide in addition to the attributes defined in EcoElement.

    Attributes
    ----------
    name : str
    Name of the economic element.

    cashflow : np.ndarray
        Cashflow of the economic element per period.

    cashflow_dis : np.ndarray
        Discounted cashflow per period.

    prj : float
        Project value (undiscounted sum of cashflows).

    dis : float
        Discounted project value.

    ann : float
        Annuity value of the economic element.

    yrl : float
        Sum of transactions during one year. Scaled from simulation results.
    """

    @property
    @abstractmethod
    def yrl(self) -> float: ...


@dataclass
class BlockElement(ABC):
    """
    Base class for all block elements.

    This abstract base class defines the common interface and properties
    that every block element in the model must provide in addition to the attributes defined in EcoElement.

    Attributes
    ----------
    name : str
    Name of the economic element.

    """

    name: str
    capex: CapexElement
    mntex: YearlyElement
    opex: YearlyElement
    crev: YearlyElement
