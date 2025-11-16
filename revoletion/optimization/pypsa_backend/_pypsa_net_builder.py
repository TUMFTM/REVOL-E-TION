import logging
import warnings

import numpy as np
import pandas as pd
import pypsa
from typing_extensions import TypeAlias

from ._utils import get_datetime_index_time_step_in_hours, normalize_datetime_index

_OptTimeSeriesArg: TypeAlias = float | pd.Series | pd.DataFrame | None

_LOGGER = logging.getLogger(__name__)


class PyPSANetworkBuilder:
    """Helper class to make it easier and more streamlined to create PyPSA neworks.

    The builder handles the temporal alignment of the input data and ensures consistent construction of each PyPSA component.
    """

    def __init__(self, datetime_index: pd.DatetimeIndex) -> None:
        self._net = pypsa.Network()

        # PyPSA does not support snapshots with TZ information.
        datetime_index = normalize_datetime_index(datetime_index)
        self._net.set_snapshots(datetime_index)  # type: ignore

        # PyPSA by default assumes a weighting of 1.0 corresponding to an timestep size of 1h.
        # However, REVOL-E-TION supports arbitrary timesteps, therefore PyPSA must be adjusted
        # according to the time step. This is done through the weighting.
        # E.g., for a datetime index with a freq of 15min the correct weighting is 0.25h.
        self._weighting = get_datetime_index_time_step_in_hours(datetime_index)
        self._net.snapshot_weightings.objective = self._weighting
        self._net.snapshot_weightings.stores = self._weighting
        self._net.snapshot_weightings.generators = self._weighting

    def add_bus(self, name: str, carrier: str | None = None) -> None:
        _LOGGER.debug(f"Adding bus '{name}': carrier={carrier}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            _ = self._net.add("Bus", name=name, carrier=carrier)

    def add_link(
        self,
        name: str,
        bus0: str,
        bus1: str,
        p_nom: int | float | None = None,
        p_nom_min: _OptTimeSeriesArg = None,
        p_nom_max: _OptTimeSeriesArg = None,
        p_nom_extendable: bool = False,
        efficiency: float | None = None,
        p_set: _OptTimeSeriesArg = None,
        p_max_pu: _OptTimeSeriesArg = None,
        p_min_pu: _OptTimeSeriesArg = None,
        marginal_cost: _OptTimeSeriesArg = None,
        capital_cost: _OptTimeSeriesArg = None,
    ) -> None:
        _LOGGER.debug(
            f"Adding link '{name}' from '{bus0}' to '{bus1}': capacity={p_nom}; invest={p_nom_extendable}; eff={efficiency}"
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            _ = self._net.add(
                "Link",
                name=name,
                bus0=bus0,
                bus1=bus1,
                p_nom=p_nom,
                p_nom_min=self._normalize_optional_timeseries_input(p_nom_min),
                p_nom_max=self._normalize_optional_timeseries_input(p_nom_max),
                p_nom_extendable=p_nom_extendable,
                efficiency=efficiency,
                p_set=self._normalize_optional_timeseries_input(p_set),
                p_max_pu=self._normalize_optional_timeseries_input(p_max_pu),
                p_min_pu=self._normalize_optional_timeseries_input(p_min_pu),
                marginal_cost=self._normalize_optional_timeseries_input(marginal_cost),
                capital_cost=self._normalize_optional_timeseries_input(capital_cost),
            )

    def add_load(
        self,
        name: str,
        bus: str,
        p_set: _OptTimeSeriesArg = None,
    ) -> None:
        _LOGGER.debug(f"Adding load '{name}' to '{bus}'")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            _ = self._net.add(
                "Load",
                name=name,
                bus=bus,
                p_set=self._normalize_optional_timeseries_input(p_set),
            )

    def add_generator(
        self,
        name: str,
        bus: str,
        p_nom: int | float | None = None,
        p_nom_min: _OptTimeSeriesArg = None,
        p_nom_max: _OptTimeSeriesArg = None,
        p_nom_extendable: bool = False,
        p_max_pu: _OptTimeSeriesArg = None,
        p_set: _OptTimeSeriesArg = None,
        marginal_cost: _OptTimeSeriesArg = None,
        capital_cost: _OptTimeSeriesArg = None,
        control: str | None = None,
        sign: float | None = None,
    ) -> None:
        _LOGGER.debug(f"Adding generator '{name}' to '{bus}': capacity={p_nom}; invest={p_nom_extendable}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            _ = self._net.add(
                "Generator",
                name=name,
                bus=bus,
                p_nom=p_nom,
                p_nom_min=self._normalize_optional_timeseries_input(p_nom_min),
                p_nom_max=self._normalize_optional_timeseries_input(p_nom_max),
                p_nom_extendable=p_nom_extendable,
                p_max_pu=self._normalize_optional_timeseries_input(p_max_pu),
                p_set=self._normalize_optional_timeseries_input(p_set),
                marginal_cost=self._normalize_optional_timeseries_input(marginal_cost),
                capital_cost=self._normalize_optional_timeseries_input(capital_cost),
                control=control,
                sign=sign,
            )

    def add_store(
        self,
        name: str,
        bus: str,
        e_nom: int | float | None = None,
        e_nom_min: _OptTimeSeriesArg = None,
        e_nom_max: _OptTimeSeriesArg = None,
        e_nom_extendable: bool = False,
        e_min_pu: _OptTimeSeriesArg = None,
        e_max_pu: _OptTimeSeriesArg = None,
        e_initial: float | None = None,
        marginal_cost: _OptTimeSeriesArg = None,
        capital_cost: _OptTimeSeriesArg = None,
        standing_loss: float | None = None,
    ) -> None:
        _LOGGER.debug(f"Adding store '{name}' to '{bus}': capacity={e_nom}; invest={e_nom_extendable}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            _ = self._net.add(
                "Store",
                name=name,
                bus=bus,
                e_nom=e_nom,
                e_nom_min=self._normalize_optional_timeseries_input(e_nom_min),
                e_nom_max=self._normalize_optional_timeseries_input(e_nom_max),
                e_nom_extendable=e_nom_extendable,
                e_min_pu=self._normalize_optional_timeseries_input(e_min_pu),
                e_max_pu=self._normalize_optional_timeseries_input(e_max_pu),
                e_initial=e_initial,
                marginal_cost=self._normalize_optional_timeseries_input(marginal_cost),
                capital_cost=self._normalize_optional_timeseries_input(capital_cost),
                standing_loss=self._scale_hourly_to_time_step(standing_loss),
            )

    def build(self) -> pypsa.Network:
        return self._net

    def _normalize_optional_timeseries_input(self, opt_timeseries_data: _OptTimeSeriesArg) -> _OptTimeSeriesArg:
        """Helper to normalize an optional timeseries argument in the `add_*` methods.

        This ensures that all timeseries data matches the format required by PyPSA.
        """
        if isinstance(opt_timeseries_data, pd.Series) or isinstance(opt_timeseries_data, pd.DataFrame):
            return opt_timeseries_data.tz_localize(tz=None).astype(dtype=np.float32)
        return opt_timeseries_data

    def _scale_hourly_to_time_step(self, value_per_hour: float | int | None) -> float | None:
        if value_per_hour is None:
            return None

        return value_per_hour * self._weighting
