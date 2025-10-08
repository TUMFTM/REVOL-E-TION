#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy as sp

from pathlib import Path

from . import utils


def lognormal_params(mean: float, stdev: float) -> tuple:
    """
    calculate lognormal parameters mu and sigma from mean and standard deviation
    """
    mu = np.log(mean**2 / np.sqrt((mean**2) + (stdev**2)))
    sig = np.sqrt(np.log(1 + (stdev**2) / (mean**2)))
    return mu, sig


class SubFleetDemand:
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
        read a usecase definition csv file and perform necessary normalization for each timeframe.
        """

        self.usecases = pd.read_csv(path_usecases, header=[0, 1], index_col=0)
        for timeframe in self.usecases.columns.levels[0]:
            self.usecases.loc[:, (timeframe, "rel_prob_norm")] = (
                self.usecases.loc[:, (timeframe, "rel_prob")] / self.usecases.loc[:, (timeframe, "rel_prob")].sum()
            )
            self.usecases.loc[:, (timeframe, "sum_dep_magn")] = (
                self.usecases.loc[:, (timeframe, "dep1_magnitude")]
                + self.usecases.loc[:, (timeframe, "dep2_magnitude")]
            )

            # catch cases where the sum of both departure magnitudes is not one
            self.usecases.loc[:, (timeframe, "dep1_magnitude")] = (
                self.usecases.loc[:, (timeframe, "dep1_magnitude")] / self.usecases.loc[:, (timeframe, "sum_dep_magn")]
            )
            self.usecases.loc[:, (timeframe, "dep2_magnitude")] = (
                self.usecases.loc[:, (timeframe, "dep2_magnitude")] / self.usecases.loc[:, (timeframe, "sum_dep_magn")]
            )

            self.usecases.drop(columns=[(timeframe, "sum_dep_magn")], inplace=True)

    def sample(self, path_timeframe_mapper: str, key_timeframe_mapper: str) -> pd.DataFrame:
        """
        generate requests dataframe from usecases & timeframes including all pre-dispatch information
        """

        self.mapper_timeframe = utils.import_module_from_path(
            module_name=path_timeframe_mapper.stem, file_path=path_timeframe_mapper
        )

        # region sample daily total requests from timeframe mapper and lognormal distribution
        daily_total = pd.DataFrame(index=pd.to_datetime(np.unique(self.dti.date)))
        daily_total["timeframe"], daily_total["demand_mean"], daily_total["demand_std"] = (
            self.mapper_timeframe.map_timeframes(daily_total, key_timeframe_mapper)
        )
        daily_total["mu"], daily_total["sigma"] = lognormal_params(
            daily_total["demand_mean"], daily_total["demand_std"]
        )
        daily_total["requests"] = daily_total.apply(
            lambda row: np.round(self.rng.lognormal(row["mu"], row["sigma"])).astype(int), axis=1
        )
        # endregion

        # region get request dates
        self.requests["date"] = pd.to_datetime(np.repeat(daily_total.index, daily_total["requests"]))
        self.requests["year"] = self.requests["date"].dt.year
        self.requests["month"] = self.requests["date"].dt.month
        self.requests["day"] = self.requests["date"].dt.day
        self.requests["timeframe"] = daily_total.loc[self.requests["date"], "timeframe"].values

        def sample_usecases(group):
            return pd.Series(
                np.random.choice(
                    self.usecases.index.values,
                    size=len(group),
                    replace=True,
                    p=self.usecases.loc[:, (group.name, "rel_prob_norm")],
                ),
                index=group.index,
            )

        self.requests["usecase"] = None
        self.requests["usecase"] = self.requests.groupby("timeframe")["usecase"].transform(sample_usecases)
        # endregion

        # region sample request times of day from usecase distribution
        def sample_time_usecase(group):
            usecase = group.name[0]
            timeframe = group.name[1]

            # always sample finer than timestep to avoid rounding errors
            timestep_hours = (pd.to_timedelta(self.dti.freq)).total_seconds() / 3600
            time_vals = np.arange(start=0, stop=24, step=timestep_hours / 100)

            mag1 = self.usecases.loc[usecase, (timeframe, "dep1_magnitude")]
            mean1 = np.median([self.usecases.loc[usecase, (timeframe, "dep1_time_mean")], 0, 24])
            std1 = np.max([self.usecases.loc[usecase, (timeframe, "dep1_time_std")], 1e-8])
            cdf1_vals = sp.stats.norm.cdf(time_vals, mean1, std1)

            mag2 = self.usecases.loc[usecase, (timeframe, "dep2_magnitude")]
            mean2 = np.median([self.usecases.loc[usecase, (timeframe, "dep2_time_mean")], 0, 24])
            std2 = np.max([self.usecases.loc[usecase, (timeframe, "dep2_time_std")], 1e-8])
            cdf2_vals = sp.stats.norm.cdf(time_vals, mean2, std2)

            cdf_vals = mag1 * cdf1_vals + mag2 * cdf2_vals
            # Generate n uniform random numbers between 0 and 1
            uniform_samples = np.random.rand(len(group))
            # Interpolate to find the samples
            time_samples = np.interp(uniform_samples, cdf_vals, time_vals)
            # round to timestep
            time_samples = np.round(time_samples / timestep_hours) * timestep_hours
            return pd.DataFrame(data=time_samples, index=group.index)

        self.requests["hour"] = (
            self.requests.groupby(["usecase", "timeframe"])
            .apply(sample_time_usecase, include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )
        self.requests["time_req"] = pd.to_datetime(self.requests[["year", "month", "day", "hour"]])
        self.requests.drop(["date", "year", "month", "day", "hour"], inplace=True, axis=1)
        self.requests["time_req"] = self.requests["time_req"].dt.tz_localize(
            self.dti.tz,
            ambiguous="NaT",  # fall
            nonexistent="shift_forward",
        )  # spring
        self.requests.dropna(axis="index", subset=["time_req"], inplace=True)
        # endregion

        self.sample_energy_demand()  # specific to type of subfleet units (battery or vehicle)

        # region sample idle time
        def sample_idle_usecase(group):
            usecase = group.name[0]
            timeframe = group.name[1]
            uc_idle_mean = self.usecases.loc[usecase, (timeframe, "idle_mean")]
            uc_idle_stdev = self.usecases.loc[usecase, (timeframe, "idle_std")]
            p1, p2 = lognormal_params(uc_idle_mean, uc_idle_stdev)
            idle = pd.to_timedelta(self.rng.lognormal(p1, p2, len(group)), unit="hour")
            return pd.Series(idle, index=group.index)

        self.requests["dtime_idle"] = None
        self.requests["dtime_idle"] = self.requests.groupby(["usecase", "timeframe"])["dtime_idle"].transform(
            sample_idle_usecase
        )
        # endregion

        # region get patience
        def get_patience_usecase(group):
            """
            groupby function
            get patience for one usecase and timeframe from usecase file
            """
            usecase = group.name[0]
            timeframe = group.name[1]
            patience = pd.to_timedelta(self.usecases.loc[usecase, (timeframe, "patience")], unit="hour")
            return pd.DataFrame({"patience_primary": [patience] * len(group)}, index=group.index)

        self.requests["dtime_patience"] = (
            self.requests.groupby(["usecase", "timeframe"])
            .apply(get_patience_usecase, include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )
        # endregion


class BatteryDemand(SubFleetDemand):
    def sample_energy_demand(self):
        """
        Sample energy requirement for each request
        """

        def sample_energy_usecase(group):
            """
            groupby function
            sample energy requirements for one usecase and timeframe.
            """
            usecase = group.name[0]
            timeframe = group.name[1]
            uc_energy_mean = self.usecases.loc[usecase, (timeframe, "energy_mean")]
            uc_energy_stdev = self.usecases.loc[usecase, (timeframe, "energy_std")]
            p1, p2 = lognormal_params(uc_energy_mean, uc_energy_stdev)
            dist = self.rng.lognormal(p1, p2, len(group))
            return pd.Series(dist, index=group.index)

        self.requests["energy_req"] = None
        self.requests["energy_req"] = self.requests.groupby(["usecase", "timeframe"])["energy_req"].transform(
            sample_energy_usecase
        )

        def calc_time_active_usecase(group):
            """
            groupby function
            calculate active time for one usecase and timeframe
            """
            usecase = group.name[0]
            timeframe = group.name[1]
            power = self.usecases.loc[usecase, (timeframe, "power_avg")]
            dtime_active = pd.to_timedelta(group["energy_req"] / power, unit="hour")
            return pd.Series(dtime_active, index=group.index)

        self.requests["dtime_active"] = (
            self.requests.groupby(["usecase", "timeframe"])
            .apply(calc_time_active_usecase, include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )


class VehicleDemand(SubFleetDemand):
    def sample_energy_demand(self):
        """
        Sample distances for each request within a VehicleFleet
        """

        def sample_distance_usecase(group):
            """
            groupby function
            sample distances for one usecase and timeframe from lognormal distribution
            """
            usecase = group.name[0]
            timeframe = group.name[1]
            uc_dist_mean = self.usecases.loc[usecase, (timeframe, "dist_mean")]
            uc_dist_stdev = self.usecases.loc[usecase, (timeframe, "dist_std")]
            p1, p2 = lognormal_params(uc_dist_mean, uc_dist_stdev)
            dist = self.rng.lognormal(p1, p2, len(group))
            return pd.Series(dist, index=group.index)

        self.requests["distance"] = np.nan
        self.requests["distance"] = self.requests.groupby(["usecase", "timeframe"])["distance"].transform(
            sample_distance_usecase
        )

        def get_consumption_speed_usecase(group):
            """
            groupby function
            get consumption and speed values for one usecase and timeframe from the usecase file
            """
            usecase = group.name[0]
            timeframe = group.name[1]
            consumption = self.usecases.loc[usecase, (timeframe, "consumption")]
            speed_avg = self.usecases.loc[usecase, (timeframe, "speed_avg")]
            return pd.DataFrame(
                data={"consumption": [consumption] * len(group), "speed_avg": [speed_avg] * len(group)},
                index=group.index,
            )

        self.requests[["consumption", "speed_avg"]] = (
            self.requests.groupby(["usecase", "timeframe"])
            .apply(get_consumption_speed_usecase, include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )

        self.requests["dtime_active"] = pd.to_timedelta(
            self.requests["distance"] / self.requests["speed_avg"], unit="hour"
        )

        self.requests["energy_req"] = self.requests["distance"] * self.requests["consumption"]
