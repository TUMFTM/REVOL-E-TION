from dataclasses import dataclass
from typing import Self, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from revoletion.blocks import blocks


@dataclass(slots=True)
class Size:
    """
    Class to store the size of a (investable) component in a block.
    This includes preexisting size, expansion size, and maximum size.
    """

    name: str
    unit: str

    preexisting: float
    expansion: float

    invest: bool
    total_max: float

    @classmethod
    def create_from_plain(
        cls,
        name: str,
        unit: str,
        preexisting: float,
        expansion: float,
        invest: bool,
        total_max: float,
    ) -> Self:
        return cls(
            name=name,
            unit=unit,
            preexisting=preexisting,
            expansion=expansion,
            invest=invest,
            total_max=total_max,
        )

    @classmethod
    def create_from_block(
        cls,
        name: str,
        block: "blocks.BaseBlock",
        unit: str = "",
        preexisting: float | None = None,
        expansion: float | None = None,
        invest: bool | None = None,
        total_max: float | None = None,
    ) -> Self:
        preexisting = float(getattr(block, f"size_preexisting_{name}", 0)) if preexisting is None else preexisting
        expansion = 0.0 if expansion is None else expansion
        invest = bool(getattr(block, f"invest_{name}", False)) if invest is None else invest
        if total_max is None:
            total_max = getattr(block, f"size_max_{name}", None)
            if total_max is None:
                total_max = np.inf
        else:
            total_max = float(total_max)
        return cls(
            name=name,
            unit=unit,
            preexisting=preexisting,
            expansion=expansion,
            invest=invest,
            total_max=total_max,
        )

    @property
    def expansion_max(self) -> float:
        # no expansion
        if not self.invest:
            return 0.0
        # restricted expansion
        elif self.total_max != np.inf:
            return max(0.0, self.total_max - self.preexisting)
        # unlimited expansion
        else:
            return np.inf

    @property
    def total(self) -> float:
        return self.preexisting + self.expansion

    @property
    def result_summary(self) -> pd.Series:
        """
        Create the result series for the summary file.
        """
        prefix = f"size_{self.name}_"
        return pd.Series(
            {
                f"{prefix}preexisting": self.preexisting,
                f"{prefix}expansion": self.expansion,
                f"{prefix}total": self.total,
                f"{prefix}invest": self.invest,
                f"{prefix}total_max": self.total_max,
                f"{prefix}expansion_max": self.expansion_max,
            }
        )

    def result_msg(self, name_block: str) -> str:
        """
        Create a message string for result_messages.
        """
        return (
            f'Optimized size of component "{self.name}" in block "{name_block}": '
            f"{self.total / 1e3:.1f} {self.unit} "
            f"(existing: {self.preexisting:.1f} {self.unit} - "
            f"expansion: {self.expansion:.1f} {self.unit})"
            if self.invest
            else ""
        )
