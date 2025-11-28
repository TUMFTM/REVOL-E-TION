#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp

from . import utils


class FleetDemand:
    """
    abstract class
    """

    def __init__(self, dti: pd.DatetimeIndex):
        self.dti = dti

        self.mapper_timeframe = None  # remains unfilled if requests is read from file
        self.usecases = None  # remains unfilled if requests is read from file
        self.requests = pd.DataFrame()  # main DataFrame for requests

        self.rng = np.random.default_rng()  # random number generator

    def from_usecases(
        self, path_usecases: str, path_timeframe_mapper: str, key_timeframe_mapper: str, path_demand: str = None
    ):
        self.read_usecase_file(path_usecases=Path(path_usecases).resolve())
        self.sample(
            path_timeframe_mapper=Path(path_timeframe_mapper).resolve(), key_timeframe_mapper=key_timeframe_mapper
        )

        if path_demand is not None:
            self.requests.to_csv(Path(path_demand).resolve())

    def from_file(self, path_demand=str, dti=None):
        """
        read in a subfleet requests csv file directly
        """
        self.requests = pd.read_csv(Path(path_demand).resolve(), index_col=0)

        self.requests["time_req"] = pd.to_datetime(self.requests["time_req"], utc=True).dt.tz_convert(self.dti.tz)
        self.requests["dtime_active"] = pd.to_timedelta(self.requests["dtime_active"])
        self.requests["dtime_idle"] = pd.to_timedelta(self.requests["dtime_idle"])
        self.requests["dtime_patience"] = pd.to_timedelta(self.requests["dtime_patience"])

        dti_filter = dti if dti is not None else self.dti
        self.requests = self.requests.loc[self.requests["time_req"].isin(dti_filter), :]

    def read_usecase_file(self, path_usecases: str) -> pd.DataFrame:
        """
        read a usecase csv file and check for normalization of mixture model weights.
        """

        self.usecases = pd.read_csv(path_usecases, header=[0, 1], index_col=[0, 1])

        self.usecases.index.names = ["usecase", "timeframe"]
        self.usecases.columns.names = ["variable", "parameter"]

        if any(self.usecases[("time_req", "weight1")] + self.usecases[("time_req", "weight2")] != 1):
            raise ValueError(f"usecase file {path_usecases}: departure time mixture weights must add to 1")

    def sample(self, path_timeframe_mapper: str, key_timeframe_mapper: str) -> pd.DataFrame:
        """
        generate requests dataframe from usecases & timeframes including all pre-dispatch information
        """

        self.mapper_timeframe = utils.import_module_from_path(
            module_name=path_timeframe_mapper.stem, file_path=path_timeframe_mapper
        )

        # region sample daily usecase requests from timeframe mapper and poisson distribution
        days = pd.DataFrame(index=pd.to_datetime(np.unique(self.dti.date)))
        days["timeframe"] = self.mapper_timeframe.map_timeframes(days, key_timeframe_mapper)

        usecase_lambdas = {
            usecase: df["demand", "lambda"].droplevel("usecase")
            for usecase, df in self.usecases.groupby(level="usecase")
        }

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
        self.requests = pd.concat(requests_dfs, ignore_index=True)
        # endregion

        # region sample request times of day from usecase distribution
        def sample_time_uctf(group):
            # always sample finer than timestep to avoid rounding errors
            timestep_hours = (pd.to_timedelta(self.dti.freq)).total_seconds() / 3600

            weights = [
                self.usecases.loc[group.name, ("time_req", "weight1")],
                self.usecases.loc[group.name, ("time_req", "weight2")],
            ]
            means = [
                self.usecases.loc[group.name, ("time_req", "mean1")],
                self.usecases.loc[group.name, ("time_req", "mean2")],
            ]
            stds = [
                self.usecases.loc[group.name, ("time_req", "std1")],
                self.usecases.loc[group.name, ("time_req", "std2")],
            ]

            # Sample from GMM
            component = np.random.choice(len(weights), size=len(group), p=weights)
            time_samples = np.random.normal(loc=np.array(means)[component], scale=np.array(stds)[component])
            # Round to timestep
            time_samples = np.round(time_samples / timestep_hours) * timestep_hours
            return pd.DataFrame(data=time_samples, index=group.index)

        self.requests["hour"] = (
            self.requests.groupby(["usecase", "timeframe"])
            .apply(sample_time_uctf, include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )

        self.requests["time_req"] = self.requests["date"] + pd.to_timedelta(self.requests["hour"], unit="h")
        self.requests.drop(["date", "hour"], inplace=True, axis=1)
        self.requests["time_req"] = self.requests["time_req"].dt.tz_localize(
            self.dti.tz,
            ambiguous="NaT",  # fall
            nonexistent="shift_forward",
        )
        self.requests.dropna(axis="index", subset=["time_req"], inplace=True)
        self.requests.sort_values(by=["time_req"], inplace=True)
        self.requests.reset_index(drop=True, inplace=True)
        # endregion

        self.sample_energy_demand()  # specific to type of subfleet units (battery or vehicle)

        # region sample idle time
        def sample_idle_uctf(group):
            p0 = self.usecases.at[group.name, ("idle", "p0")]
            a = self.usecases.at[group.name, ("idle", "a")]
            c = self.usecases.at[group.name, ("idle", "c")]
            scale = self.usecases.at[group.name, ("idle", "scale")]
            return pd.Series(
                pd.to_timedelta(
                    sp.stats.gengamma.rvs(a=a, c=c, scale=scale, size=len(group)) * (1 - p0) + p0, unit="hour"
                ),
                index=group.index,
            )

        self.requests["dtime_idle"] = None
        self.requests["dtime_idle"] = self.requests.groupby(["usecase", "timeframe"])["dtime_idle"].transform(
            sample_idle_uctf
        )
        # endregion

        # region get patience
        def get_patience_uctf(group):
            """
            groupby function
            get patience for one usecase and timeframe from usecase file
            """
            patience = pd.to_timedelta(self.usecases.loc[group.name, ("patience", "value")], unit="hour")
            return pd.DataFrame({"patience_primary": [patience] * len(group)}, index=group.index)

        self.requests["dtime_patience"] = (
            self.requests.groupby(["usecase", "timeframe"])
            .apply(get_patience_uctf, include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )
        # endregion


class BatteryFleetDemand(FleetDemand):
    def sample_energy_demand(self):
        """
        Sample energy requirement for each request
        """

        def sample_energy_uctf(group):
            """
            groupby function
            sample energy requirements for one usecase and timeframe.
            """
            return pd.Series(
                np.random.lognormal(
                    mean=self.usecases.loc[group.name, ("energy", "mu")],
                    sigma=self.usecases.loc[group.name, ("energy", "sigma")],
                    size=len(group),
                ),
                index=group.index,
            )

        self.requests["energy_req"] = None
        self.requests["energy_req"] = self.requests.groupby(["usecase", "timeframe"])["energy_req"].transform(
            sample_energy_uctf
        )

        def calc_time_active_uctf(group):
            """
            groupby function
            calculate active time for one usecase and timeframe
            """
            power = self.usecases.loc[group.name, ("power", "value")]
            dtime_active = pd.to_timedelta(group["energy_req"] / power, unit="hour")
            return pd.Series(dtime_active, index=group.index)

        self.requests["dtime_active"] = (
            self.requests.groupby(["usecase", "timeframe"])
            .apply(calc_time_active_uctf, include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )


class VehicleFleetDemand(FleetDemand):
    def sample_energy_demand(self):
        """
        Sample distances for each request within a VehicleFleet
        """

        def sample_distance_uctf(group):
            """
            groupby function
            sample distances for one usecase and timeframe from lognormal distribution
            """
            return pd.Series(
                np.random.lognormal(
                    mean=self.usecases.loc[group.name, ("dist", "mu")],
                    sigma=self.usecases.loc[group.name, ("dist", "sigma")],
                    size=len(group),
                ),
                index=group.index,
            )

        self.requests["distance"] = np.nan
        self.requests["distance"] = self.requests.groupby(["usecase", "timeframe"])["distance"].transform(
            sample_distance_uctf
        )

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

        self.requests[["consumption", "speed", "subfleets"]] = (
            self.requests.groupby(["usecase", "timeframe"])
            .apply(get_params_uctf, include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )

        self.requests["dtime_active"] = pd.to_timedelta(self.requests["distance"] / self.requests["speed"], unit="hour")

        self.requests["energy_req"] = self.requests["distance"] * self.requests["consumption"]
