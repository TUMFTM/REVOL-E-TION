#!/usr/bin/env python3
import ast
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from . import stochastics, utils


def parse_entry(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed


# ToDo: FleetDemand currently does not need to  be a class as it's just a pd.DataFrame.
#  -> convert to functions (read_demand_file + sample_demand)
#  -> for sampling use object of class Sampler (BatteryFleetSampler + VehicleFleetSampler)
#  -> think about: dataclass UseCase
class FleetDemand(ABC):
    """
    abstract class
    """

    def __init__(self, dti: pd.DatetimeIndex):
        self.dti = dti

        self.mapper_timeframe = None  # remains unfilled if requests is read from file
        self.usecases = None  # remains unfilled if requests is read from file
        self.requests = pd.DataFrame()  # main DataFrame for requests

    def from_usecases(
        self,
        path_usecases: Path,
        path_timeframe_mapper: Path,
        key_timeframe_mapper: str,
        path_demand: Path = None,
        subfleets: list = None,
    ):
        self.usecases = self.read_usecase_file(path_usecases=path_usecases)

        if (subfleets is not None) and (("subfleets", "list") in self.usecases.columns):
            subfleets_set = set(subfleets)
            self.usecases = self.usecases[
                self.usecases[("subfleets", "list")].apply(
                    lambda usecase_subfleets: any(item in subfleets_set for item in usecase_subfleets)
                )
            ]

        self.mapper_timeframe = self.get_timeframe_mapper(path_timeframe_mapper=path_timeframe_mapper)
        self.requests = self.sample(key_timeframe_mapper=key_timeframe_mapper)

        if path_demand is not None:
            self.requests.to_feather(path_demand)

    def from_file(self, path_demand: Path, dti=None):
        """
        read in a subfleet requests file (CSV or feather) directly
        """
        if Path(path_demand).suffix == ".feather":
            # feather preserves the native dtypes, so the timedeltas and the (numpy array) subfleets only
            # need normalizing to match the CSV path below
            requests = pd.read_feather(path_demand)
            if "subfleets" in requests.columns:
                requests["subfleets"] = requests["subfleets"].apply(list)
        else:
            requests = pd.read_csv(path_demand, index_col=0)
            requests["dtime_active"] = pd.to_timedelta(requests["dtime_active"])
            requests["dtime_idle"] = pd.to_timedelta(requests["dtime_idle"])
            requests["dtime_patience"] = pd.to_timedelta(requests["dtime_patience"])
            if "subfleets" in requests.columns:
                requests["subfleets"] = requests["subfleets"].apply(parse_entry)

        requests["time_req"] = pd.to_datetime(requests["time_req"], utc=True).dt.tz_convert(self.dti.tz)

        dti_filter = dti if dti is not None else self.dti
        self.requests = requests.loc[requests["time_req"].isin(dti_filter), :]

    @staticmethod
    def read_usecase_file(path_usecases: Path) -> pd.DataFrame:
        """
        read a usecase csv file and check for normalization of mixture model weights.
        """

        usecases = pd.read_csv(path_usecases, header=[0, 1], index_col=[0, 1])

        if ("subfleets", "list") in usecases.columns:
            usecases[("subfleets", "list")] = usecases[("subfleets", "list")].apply(parse_entry)

        usecases.index.names = ["usecase", "timeframe"]
        usecases.columns.names = ["variable", "parameter"]

        weight_sum = usecases[("time_req", "weight1")] + usecases[("time_req", "weight2")]
        usecases[("time_req", "weight1")] /= weight_sum
        usecases[("time_req", "weight2")] /= weight_sum

        return usecases

    @staticmethod
    def get_timeframe_mapper(path_timeframe_mapper: Path):
        return utils.import_module_from_path(module_name=path_timeframe_mapper.stem, file_path=path_timeframe_mapper)

    @abstractmethod
    def sample_energy_demand(self, requests: pd.DataFrame) -> pd.DataFrame: ...

    def sample(self, key_timeframe_mapper: str) -> pd.DataFrame:
        """
        generate requests dataframe from usecases & timeframes including all pre-dispatch information
        """

        # region sample daily usecase requests from timeframe mapper and poisson distribution
        days = pd.DataFrame(index=pd.to_datetime(np.unique(self.dti.date)))
        days["timeframe"] = self.mapper_timeframe.map_timeframes(days, key_timeframe_mapper)

        usecase_lambdas = {
            usecase: df["demand", "lambda"].droplevel("usecase")
            for usecase, df in self.usecases.groupby(level="usecase")
        }

        if not usecase_lambdas:  # no usecases defined
            requests = pd.DataFrame(
                columns=[
                    "date",
                    "usecase",
                    "timeframe",
                    "time_req",
                    "energy_req",
                    "dtime_active",
                    "dtime_idle",
                    "dtime_patience",
                ]
            )
            return requests

        for usecase, lambdas in usecase_lambdas.items():
            lam_values = days["timeframe"].map(lambdas).fillna(0)
            days[f"{usecase}"] = np.random.poisson(lam=lam_values)
        # endregion

        # region fill requests dataframe
        requests_dfs = []
        for usecase in usecase_lambdas:
            dates = np.repeat(days.index.values, days[usecase].values)
            timeframes = np.repeat(days["timeframe"].values, days[usecase].values)
            requests_uc = pd.DataFrame({"date": dates, "usecase": usecase, "timeframe": timeframes})
            requests_dfs.append(requests_uc)

        requests = pd.concat(requests_dfs, ignore_index=True)
        # endregion

        # region sample request times of day from usecase distribution
        def sample_time_uctf(group):
            # always sample finer than timestep to avoid rounding errors
            timestep_hours = (pd.to_timedelta(self.dti.freq)).total_seconds() / 3600

            model = stochastics.DepartureDistribution.from_mean_std(
                weight1=self.usecases.at[group.name, ("time_req", "weight1")],
                weight2=self.usecases.at[group.name, ("time_req", "weight2")],
                mean1=self.usecases.at[group.name, ("time_req", "mean1")],
                mean2=self.usecases.at[group.name, ("time_req", "mean2")],
                std1=self.usecases.at[group.name, ("time_req", "std1")],
                std2=self.usecases.at[group.name, ("time_req", "std2")],
            )

            samples = model.sample(size=len(group))
            time_samples = np.round(samples / timestep_hours) * timestep_hours  # Round to timestep
            return pd.DataFrame({"hour": time_samples}, index=group.index)

        requests["hour"] = requests.groupby(["usecase", "timeframe"], group_keys=False).apply(sample_time_uctf)

        requests["time_req"] = requests["date"] + pd.to_timedelta(requests["hour"], unit="h")
        requests.drop(["date", "hour"], inplace=True, axis=1)
        requests["time_req"] = requests["time_req"].dt.tz_localize(
            self.dti.tz,
            ambiguous="NaT",  # fall
            nonexistent="shift_forward",
        )
        requests.dropna(axis="index", subset=["time_req"], inplace=True)
        requests.sort_values(by=["time_req"], inplace=True)
        requests.reset_index(drop=True, inplace=True)
        # endregion

        requests = self.sample_energy_demand(
            requests=requests
        )  # specific to type of subfleet units (battery or vehicle)

        # region sample idle time
        def sample_idle_uctf(group):
            distribution = stochastics.IdleDistribution(
                p0=self.usecases.at[group.name, ("idle", "p0")],
                a=self.usecases.at[group.name, ("idle", "a")],
                c=self.usecases.at[group.name, ("idle", "c")],
                scale=self.usecases.at[group.name, ("idle", "scale")],
            )

            return pd.Series(
                pd.to_timedelta(distribution.sample(size=len(group)), unit="hour"),
                index=group.index,
            )

        requests["dtime_idle"] = None
        requests["dtime_idle"] = requests.groupby(["usecase", "timeframe"])["dtime_idle"].transform(sample_idle_uctf)
        # endregion

        # region get patience
        def get_patience_uctf(group):
            """
            groupby function
            get patience for one usecase and timeframe from usecase file
            """
            patience = pd.to_timedelta(self.usecases.loc[group.name, ("patience", "value")], unit="hour")
            return pd.DataFrame({"patience_primary": [patience] * len(group)}, index=group.index)

        requests["dtime_patience"] = (
            requests.groupby(["usecase", "timeframe"])
            .apply(get_patience_uctf, include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )
        # endregion

        return requests


class BatteryFleetDemand(FleetDemand):
    def sample_energy_demand(self, requests: pd.DataFrame) -> pd.DataFrame:
        """
        Sample energy requirement for each request
        """

        def sample_energy_uctf(group):
            """
            groupby function
            sample energy requirements for one usecase and timeframe.
            """
            distribution = stochastics.EnergyDistribution(
                alpha=self.usecases.at[group.name, ("energy", "alpha")],
                beta=self.usecases.at[group.name, ("energy", "beta")],
                capacity=self.usecases.at[group.name, ("energy", "capacity")],
            )
            return pd.Series(
                distribution.sample(size=len(group)),
                index=group.index,
            )

        requests["energy_req"] = None
        requests["energy_req"] = requests.groupby(["usecase", "timeframe"])["energy_req"].transform(sample_energy_uctf)

        def calc_time_active_uctf(group):
            """
            groupby function
            calculate active time for one usecase and timeframe
            """
            power = self.usecases.loc[group.name, ("power", "value")]
            dtime_active = pd.to_timedelta(group["energy_req"] / power, unit="hour")
            return pd.Series(dtime_active, index=group.index)

        requests["dtime_active"] = (
            requests.groupby(["usecase", "timeframe"])
            .apply(calc_time_active_uctf, include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )

        return requests


class VehicleFleetDemand(FleetDemand):
    def sample_energy_demand(self, requests: pd.DataFrame) -> pd.DataFrame:
        """
        Sample distances for each request within a VehicleFleet
        """

        def sample_distance_uctf(group):
            """
            groupby function
            sample distances for one usecase and timeframe from lognormal distribution
            """
            distribution = stochastics.DistanceDistribution.from_mu_sigma(
                mu=self.usecases.at[group.name, ("dist", "mu")], sigma=self.usecases.at[group.name, ("dist", "sigma")]
            )
            return pd.Series(
                distribution.sample(size=len(group)),
                index=group.index,
            )

        requests["distance"] = np.nan
        requests["distance"] = requests.groupby(["usecase", "timeframe"])["distance"].transform(sample_distance_uctf)

        def get_params_uctf(group):
            """
            groupby function
            get consumption and speed values for one usecase and timeframe from the usecase file
            """
            consumption = self.usecases.loc[group.name, ("consumption", "value")]
            speed = self.usecases.loc[group.name, ("speed", "value")]
            subfleets = self.usecases.loc[group.name, ("subfleets", "list")]
            return pd.DataFrame(
                data={
                    "consumption": [consumption] * len(group),
                    "speed": [speed] * len(group),
                    "subfleets": [subfleets] * len(group),
                },
                index=group.index,
            )

        requests[["consumption", "speed", "subfleets"]] = (
            requests.groupby(["usecase", "timeframe"])
            .apply(get_params_uctf, include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )

        requests["dtime_active"] = pd.to_timedelta(requests["distance"] / requests["speed"], unit="hour")

        requests["energy_req"] = requests["distance"] * requests["consumption"]

        return requests
