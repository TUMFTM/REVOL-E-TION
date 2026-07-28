#!/usr/bin/env python3

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import numpy as np
import pandas as pd
import windpowerlib

from revoletion import battery as bat
from revoletion import data_manager, energy, mobility, models, peak_periods, size, time, utils
from revoletion import economics as eco
from revoletion import scenario as scn

if TYPE_CHECKING:
    import datetime


class BlockScenarioInterface(ABC):
    @abstractmethod
    def pre_scenario(self, **kwargs) -> None:
        """
        Trigger actions to be executed after all inits.
        """
        ...

    @abstractmethod
    def post_scenario(self) -> None:
        """
        Trigger actions to be executed after the scenario has been run.
        """
        ...


class BaseBlock(BlockScenarioInterface, ABC):
    _SIZE_NAMES = []
    _FLOW_NAMES = []
    _STATE_NAMES = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Merge the _FLOW_NAMES, _STATE_NAMES, and _SIZE_NAMES from all parent classes and own definition
        for attr in ("_FLOW_NAMES", "_STATE_NAMES", "_SIZE_NAMES"):
            accumulated = []

            # Merge from direct parents only (as they already contain the merged lists of their parents)
            for base in cls.__bases__:
                if hasattr(base, attr):
                    accumulated.extend(getattr(base, attr))

            # Add own definition (if any)
            accumulated.extend(cls.__dict__.get(attr, []))

            setattr(cls, attr, accumulated)

    def init_pois(self):
        # add a new POI to block.pois
        pass

    def __init__(
        self,
        name: str,
        scenario: scn.Scenario,
        params: dict = None,
        parent: BaseBlock | scn.Scenario = None,
        **kwargs,
    ):
        """
        Initialize (Sub)Block object with attributes and data structures
        """

        self.name = name
        self.scenario = scenario
        self.parent = parent

        self.classname = self.__class__.__name__  # get name of class
        self.top_level_block = True if self.parent is self.scenario else False  # distinguish top level blocks/subblocks

        self.register_block()

        # region set attributes from scenario file or parent
        params = params if params is not None else self.scenario.parameters.loc[self.name]
        for key, value in params.items():
            setattr(self, key, value)
        # endregion

        # preprocessing of invest/sizes which are set to equal will be set to the same value
        self.expansion_equal = False
        self.params_preprocessing()

        self.sizes = {
            name: size.Size.create_from_block(name=name, block=self, unit=unit) for name, unit in self._SIZE_NAMES
        }

        self.flows = pd.DataFrame(
            index=self.scenario.times.sim.dti,
            columns=self._FLOW_NAMES,
            data=0.0,
            dtype=float,
        )

        self.states = pd.DataFrame(
            index=self.scenario.times.sim.dti_extd,
            columns=self._STATE_NAMES,
            data=0.0,
            dtype=float,
        )

        self.pois = {}
        self.init_pois()

        self.aggregator = eco.Aggregator(name=self.name, prj_duration_yrs=self.scenario.eco_params.prj_duration_yrs)
        for poi in self.pois.values():
            if poi.aggregate:
                self.aggregator.add_block(poi)
        self.parent.aggregator.add_block(self.aggregator)

        self.energies = {
            flow: energy.EnergyEvaluator(name=flow, eco=self.scenario.eco_params) for flow in self._FLOW_NAMES
        }

        # accumulate invest costs caused by preexisting components in scenario
        for poi in self.pois.values():
            size_obj = self.sizes.get(poi.name_size, None)
            size_preexisting = size_obj.preexisting if size_obj else None
            self.scenario.capex_preexisting_considered += poi.get_invest_preexisting(size_preexisting=size_preexisting)

        # region initialize data structures
        self.subblocks = dict()
        # endregion

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"

    def register_block(self):
        for cls in self.__class__.__mro__:
            if cls in (object, ABC, BlockScenarioInterface):
                continue
            name_class = cls.__name__
            # Initialize per-class registry if not present and register self
            self.scenario.block_registry.setdefault(name_class, {})[self.name] = self

        if self.top_level_block:
            self.scenario.block_registry.setdefault("TopLevelBlock", {})[self.name] = self
        else:  # is subblock
            self.parent.subblocks[self.name] = self

    def params_preprocessing(self):
        pass

    def init_equalizable_variables(self, name_vars: list):
        name_var1, name_var2 = name_vars
        if (getattr(self, name_var1) == "equal") and (getattr(self, name_var2) == "equal"):
            error_msg = (
                f'"{self.name}" parameters {name_var1} and {name_var2} were both set to equal.'
                f' Maximum one of these variables is allowed to be set to "equal"'
            )
            self.scenario.logger.error(error_msg)
        elif getattr(self, name_var1) == "equal":
            setattr(self, name_var1, getattr(self, name_var2))
        elif getattr(self, name_var2) == "equal":
            setattr(self, name_var2, getattr(self, name_var1))

    @override
    def pre_scenario(self, **kwargs):
        """
        trigger actions to be executed after all inits
        """
        for subblock in self.subblocks.values():
            subblock.pre_scenario(**kwargs)

    @override
    def post_scenario(self):
        for subblock in self.subblocks.values():
            subblock.post_scenario()

        # calculate results
        self.calc_results_economics()

    def _build_poi_evaluation_kwargs(self, poi: eco.POI, **kwargs) -> dict[str, Any]:
        if poi.name_size is not None:
            size_obj = self.sizes[poi.name_size]
            size_preexisting = size_obj.preexisting
            size_expansion = size_obj.expansion
        else:
            size_preexisting = None
            size_expansion = None

        flow_name = poi.name_flow
        if flow_name:
            flow = self.flows[flow_name].loc[self.scenario.times.eval.dti]
        else:
            flow = None

        return {
            "size_preexisting": size_preexisting,
            "size_expansion": size_expansion,
            "power": flow,
        }

    def calc_results_economics(self, **kwargs):
        # calculate economic results
        for poi in self.pois.values():
            poi.evaluate(**self._build_poi_evaluation_kwargs(poi), **kwargs)

        self.aggregator.aggregate()


class NonElectricBlock(BaseBlock): ...


class ElectricBlock(BaseBlock, ABC):
    _FLOW_NAMES = ["total"]

    def __init__(
        self,
        name: str,
        scenario: scn.Scenario,
        flow_apriori_names: list = None,
        params: dict = None,
        parent: BaseBlock | scn.Scenario = None,
        **kwargs,
    ):
        self.power_circles = []

        super().__init__(name=name, scenario=scenario, params=params, parent=parent, **kwargs)

        # empty list not possible as default argument as it is mutable
        flow_apriori_names = flow_apriori_names if flow_apriori_names is not None else []

        self.components = {}
        self.bus_connected = None

        # ToDo: (1) remove flow_apriori_names and use flow names instead
        #       (2) remove flows_apriori and use flows instead to save memory
        self.flows_apriori = pd.DataFrame(
            index=self.scenario.times.sim.dti,
            columns=flow_apriori_names,
            dtype="float64",
        )

        self.eff = dict()
        self.initialize_efficiencies()

    def initialize_efficiencies(self):
        for key in list(self.__dict__.keys()):  # use list() to safely modify the dict (delattr) while iterating
            if key.startswith("eff_"):
                self.eff[re.sub(r"^[^_]+_", "", key)] = getattr(self, key)
                delattr(self, key)

    def post_scenario(self):
        self.calc_results_flows()
        self.calc_results_energies()

        super().post_scenario()

    def calc_results_flows(self):
        # total flow calculation is duplicated in StorageBlock
        self.flows["total"] = self.flows.get(key="out", default=0) - self.flows.get(key="in", default=0)

        # detect circular flows
        for power_circle in self.power_circles:
            self.detect_circular_flows(flow1=power_circle[0], flow2=power_circle[1])

    def detect_circular_flows(self, flow1: str, flow2: str):
        col1 = self.flows[flow1].to_numpy()
        col2 = self.flows[flow2].to_numpy()
        circular = np.minimum(col1, col2)
        if np.any(circular):
            self.flows[f"circular_{flow1}_{flow2}"] = circular
            self.scenario.logger.warning(
                f'Block "{self.name}" - circular flow for flows {flow1} and {flow2} - check energy results'
            )

    def calc_results_energies(self):
        """
        post scenario method
        process flows and calculate energies from flows
        """
        for e in self.energies.values():
            e.evaluate(self.flows.loc[self.scenario.times.eval.dti, e.name])


class SourceBlock(ElectricBlock, ABC):
    def calc_results_energies(self):
        super().calc_results_energies()
        self.scenario.energies["sources"].add_energy(self.energies["total"])


class SinkBlock(ElectricBlock, ABC):
    def calc_results_energies(self):
        super().calc_results_energies()
        self.scenario.energies["sinks"].add_energy(self.energies["total"])


class SystemCore(ElectricBlock):
    _SIZE_NAMES = [("acdc", "kW"), ("dcac", "kW")]
    _FLOW_NAMES = ["acdc", "dcac", "deficit_ac", "deficit_dc"]

    _SYSTEMS_DEFICIT = ["ac", "dc"]

    def init_pois(self):
        super().init_pois()

        self.pois["acdc"] = eco.POI.create(
            name="acdc",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex=eco.CapexParams(
                spec=self.capex_spec,
                consider_preexisting=self.capex_acdc_preexisting,
                ls=self.ls,
                ccr=self.ccr,
            ),
            mntex=eco.MntexParams(spec=self.mntex_spec),
            opex=eco.OpexParams(spec_energy=self.opex_spec),
            name_size="acdc",
            name_flow="acdc",
        )

        self.pois["dcac"] = eco.POI.create(
            name="dcac",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex=eco.CapexParams(
                spec=self.capex_spec,
                consider_preexisting=self.capex_dcac_preexisting,
                ls=self.ls,
                ccr=self.ccr,
            ),
            mntex=eco.MntexParams(spec=self.mntex_spec),
            opex=eco.OpexParams(spec_energy=self.opex_spec),
            name_size="dcac",
            name_flow="dcac",
        )

        # The deficit sources are unlimited in power and free of capex and mntex. Their opex only serves as a
        # penalty steering the optimizer away from them and is therefore excluded from the aggregated results.
        for system in self._SYSTEMS_DEFICIT:
            self.pois[f"deficit_{system}"] = eco.POI.create(
                name=f"deficit_{system}",
                eco=self.scenario.eco_params,
                data_dir=self.scenario.paths.input,
                opex=eco.OpexParams(spec_energy=self.opex_spec_deficit),
                name_flow=f"deficit_{system}",
                aggregate=False,
            )

        self.power_circles.append(("acdc", "dcac"))

    def __init__(self, name: str, scenario, **kwargs):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=None,
            parent=scenario,
            **kwargs,
        )

    def params_preprocessing(self):
        # opex_spec_deficit is optional in the scenario file
        if not hasattr(self, "opex_spec_deficit"):
            self.opex_spec_deficit = models.OPEX_SPEC_DEFICIT_DEFAULT

        self.expansion_equal = True if self.invest_acdc == "equal" or self.invest_dcac == "equal" else False

        self.init_equalizable_variables(name_vars=["invest_acdc", "invest_dcac"])
        self.init_equalizable_variables(name_vars=["size_acdc_preexisting", "size_dcac_preexisting"])
        self.init_equalizable_variables(name_vars=["size_acdc_max", "size_dcac_max"])

    def calc_results_flows(self):
        """
        post scenario method
        """
        super().calc_results_flows()

    def calc_results_energies(self):
        """
        post scenario method
        """
        super().calc_results_energies()

        for system in self._SYSTEMS_DEFICIT:
            energy_deficit = self.energies[f"deficit_{system}"].eval
            if energy_deficit > 0:
                self.scenario.logger.warning(
                    f'Block "{self.name}" - {energy_deficit / 1e3:.1f} kWh of energy drawn from the '
                    f"{system.upper()} deficit source - the energy system cannot cover its demand"
                )


class RenewableSource(SourceBlock, ABC):
    _SIZE_NAMES = [("block", "kWp")]
    _FLOW_NAMES = ["out", "curt", "pot"]

    def init_pois(self):
        super().init_pois()
        self.pois["block"] = eco.POI.create(
            name="block",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex=eco.CapexParams(
                spec=self.capex_spec,
                consider_preexisting=self.capex_block_preexisting,
                ls=self.ls,
                ccr=self.ccr,
            ),
            mntex=eco.MntexParams(spec=self.mntex_spec),
            opex=eco.OpexParams(spec_energy=self.opex_spec),
            name_size="block",
            name_flow="out",
        )

        self.pois["curt"] = eco.POI.create(
            name="curt",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            name_flow="curt",
        )

        self.pois["pot"] = eco.POI.create(
            name="pot",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            name_flow="pot",
        )

    def __init__(self, name: str, scenario: scn.Scenario, **kwargs):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=None,
            parent=scenario,
            **kwargs,
        )
        self.data = None  # todo move to a priori flows (except for wind speed and ambient temp)
        self.get_ts_data()

        self.share_curtailment = None

    @abstractmethod
    def get_ts_data(self): ...

    def calc_results_energies(self):
        super().calc_results_energies()

        e_pot = self.energies["pot"].eval == 0
        if e_pot == 0:
            self.share_curtailment = np.nan
        else:
            self.share_curtailment = self.energies["curt"].eval / e_pot

        # aggregate results in scenario.energies
        self.scenario.energies["renewable_act"].add_energy(self.energies["out"])
        self.scenario.energies["renewable_pot"].add_energy(self.energies["pot"])
        self.scenario.energies["renewable_curt"].add_energy(self.energies["curt"])


class PVSource(RenewableSource):
    def get_ts_data(self):
        """
        pre scenario (init) method
        Get potential power profile from API or file, each either from Solcast or PVGIS
        """

        array = data_manager.PvArray(
            tracking_type=getattr(self, "trackingtype", 0),
            mounting_place=getattr(self, "mountingplace", "free"),
            type_cell=getattr(self, "type_cell", None),
            rad_database=getattr(self, "database", None),
            tilt=getattr(self, "tilt", None),
            azimuth=getattr(self, "azimuth", None),
            horizon_custom=getattr(self, "horizon_custom", None),
        )

        manager = data_manager.DataManager(location=self.scenario.location, logger=self.scenario.logger, array=array)

        try:
            data_source = data_manager.DataSource(self.data_source)
        except ValueError as e:
            raise RuntimeError(f"Failed to retrieve timeseries data for block {self.name}") from e

        if self.filename is not None:
            path_input_file = self.scenario.paths.input / utils.set_extension(
                filename=self.filename, default_extension=".csv"
            )
        else:
            path_input_file = None

        try:
            self.data = manager.get_data(
                data_source,
                file_path=path_input_file,
                timeframe=self.scenario.times.sim,
                api_key=self.scenario.settings.key_solcast_api,
            )

            if getattr(self, "temp_scn", False):
                self.scenario.temp_air = self.data["temp_air"].copy()

        except data_manager.DataProviderError as e:
            raise RuntimeError(f"Failed to retrieve timeseries data for block {self.name}") from e


class WindSource(RenewableSource):
    def get_ts_data(self):
        """pre scenario (init) method
        get potential power profile from PVSource block or file
        """
        if self.data_source in self.scenario.block_registry.get("TopLevelBlock", {}).keys():
            # region get data from PVSource block
            self.data = self.scenario.block_registry.get("TopLevelBlock", {})[self.data_source].data.copy()
            self.data["speed_wind_adj"] = windpowerlib.wind_speed.hellman(self.data["speed_wind"], 10, self.height)

            path_turbine_data_file = self.scenario.paths.data_persist / "turbine_data.pkl"
            turbine_data = pd.read_pickle(path_turbine_data_file)
            # smallest fully filled wind turbine in dataseta as per June 2024
            turbine_data = turbine_data.loc[turbine_data["turbine_type"] == "E-53/800"].reset_index()

            self.data["power_original"] = windpowerlib.power_output.power_curve(
                wind_speed=self.data["speed_wind_adj"],
                power_curve_wind_speeds=ast.literal_eval(turbine_data.loc[0, "power_curve_wind_speeds"]),
                power_curve_values=ast.literal_eval(turbine_data.loc[0, "power_curve_values"]),
                density_correction=False,
            )
            self.data["power_spec"] = self.data["power_original"] / turbine_data.loc[0, "nominal_power"]
            # endregion
        elif self.data_source == "file":
            # region get data from file
            try:
                self.data = utils.read_timeseries(
                    path_input_file=(
                        self.scenario.paths.input
                        / utils.set_extension(filename=self.filename, default_extension=".csv")
                    ),
                    timezone=self.scenario.location.timezone,
                    resampling_dti=self.scenario.times.sim.dti,
                )
            except IndexError as exc:
                raise IndexError(f"Failed to load timeseries data for block {self.name}: {exc}")
        else:
            raise ValueError(f"Scenario {self.scenario.name} - Block {self.name}: No usable data input specified")

        if not self.scenario.settings.largescalemode:
            # feather does not serialize a non-default index, so move the DatetimeIndex into a column
            self.data.reset_index().to_feather(
                self.scenario.paths.create_result_path(suffix=f"{self.scenario.name}_{self.name}_log.feather")
            )


class FixedDemand(SinkBlock):
    _FLOW_NAMES = ["in"]
    _SLP_IDS = ["h0", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "l0", "l1", "l2", "h25", "g25", "l25", "s25", "p25"]

    def init_pois(self):
        super().init_pois()
        self.pois["block"] = eco.POI.create(
            name="block",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex=eco.CapexParams(
                fix=self.capex_fix_metering,
                consider_preexisting=self.capex_metering_preexisting,
            ),
            mntex=eco.MntexParams(fix=self.mntex_fix_metering),
            crev=eco.CrevParams(spec_energy=self.crev_spec),
            name_flow="in",
        )

    def __init__(self, name: str, scenario, **kwargs):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=["demand"],
            params=None,
            parent=scenario,
            **kwargs,
        )

        self.get_flows_apriori()

    @staticmethod
    def get_slp(
        slp_id: str,
        ts_start: pd.Timestamp,
        ts_end: pd.Timestamp,
        path_data_file: Path,
        holiday_dates: list[datetime.date] = None,
    ):
        slp_id = slp_id.upper()
        if slp_id not in [
            "H0",
            "G0",
            "G1",
            "G2",
            "G3",
            "G4",
            "G5",
            "G6",
            "L0",
            "L1",
            "L2",
            "L3",
            "H25",
            "G25",
            "L25",
            "S25",
            "P25",
        ]:
            raise ValueError(f"SLP '{slp_id}' is not recognized.")

        if not all(ts.tz for ts in [ts_start, ts_end]):
            raise ValueError("Timestamps must be timezone-aware.")

        if holiday_dates is None:
            holiday_dates = []

        # read SLP data, do not set index here, as we need to convert time column first
        data = pd.read_csv(path_data_file, index_col=[])

        # convert time column to time objects (only time without date)
        data["time"] = pd.to_datetime(data["time"], format="%H:%M").dt.time

        # set multi-index
        data.set_index(["profile", "period", "day", "time"], inplace=True)

        # use a fixed frequency of 15 minutes for the timeseries generation as the SLPs are given with that frequency
        freq_slp = "15min"
        slp_dti = pd.DatetimeIndex(
            pd.date_range(
                start=ts_start.floor(freq_slp),
                end=ts_end.ceil(freq_slp),
                freq=freq_slp,
            )
        )

        month = slp_dti.month
        day = slp_dti.day

        # vectorized period assignment
        if slp_id in ["H25", "G25", "L25", "S25", "P25"]:
            # Use month abbreviation for these profiles
            period = pd.Series(slp_dti.strftime("%b"), index=slp_dti)
        else:
            # Use Winter/Summer/Transition logic

            period = pd.Series(index=slp_dti, dtype="object")

            # Winter: Nov 1 - Mar 20
            period[((month >= 11) | (month <= 3)) & ~((month == 3) & (day > 20))] = "Winter"

            # Summer: May 15 - Sep 14
            period[((month > 5) | ((month == 5) & (day >= 15))) & ((month < 9) | ((month == 9) & (day <= 14)))] = (
                "Summer"
            )

            # Transition: Mar 21 - May 14 and Sep 15 - Oct 31 (-> everything else)
            period.fillna("Transition", inplace=True)

        # vectorized daytype
        dow = slp_dti.weekday
        daytype = pd.Series("Workday", index=slp_dti)
        # treat Christmas Eve and New Year's Eve as Saturdays, will be overwritten if they are Sundays
        daytype[dow == 5 | ((month.isin([12])) & (day.isin([24, 31])))] = "Saturday"
        # to use isin for holidays: remove timezone info and normalize to midnight
        daytype[dow == 6 | slp_dti.tz_localize(None).normalize().isin(pd.to_datetime(holiday_dates))] = "Sunday"

        lookup_index = pd.MultiIndex.from_arrays(
            arrays=[[slp_id] * len(slp_dti), period.values, daytype.values, slp_dti.time],
            names=data.index.names,
        )

        # use reindex to execute lookup
        slp_timeseries = data.reindex(lookup_index).set_axis(slp_dti)

        # for private households use dynamic correction as stated in VDEW manual
        if slp_id in ["H0", "H25", "P25", "S25"]:
            factor = np.polyval(p=[-3.92e-10, 3.2e-7, -7.02e-5, 2.1e-3, 1.24], x=slp_dti.dayofyear)
            slp_timeseries = slp_timeseries.mul(factor, axis=0)

        return slp_timeseries

    def get_flows_apriori(self):
        self.flows_apriori.index = (
            self.scenario.times.sim.dti
        )  # ToDo: Why needs this to be set explicitly? Should be done in init()

        if self.load_profile in self._SLP_IDS:
            data = self.get_slp(
                slp_id=self.load_profile,
                ts_start=self.scenario.times.sim.start,
                ts_end=self.scenario.times.sim.end,
                holiday_dates=self.scenario.holiday_dates,
                path_data_file=self.scenario.paths.data_persist / "slp_bdew.csv",
            )

            # scale load profile (given for consumption of 1MWh per year) to specified yearly consumption
            # this calculation leads to small deviations from the specified yearly consumption due to varying holidays and
            # leap years, but is the correct way as stated by the VDEW manual
            data *= self.consumption_yrl / 1e6

            # resample to simulation time step
            self.flows_apriori["demand"] = data.resample(self.scenario.timestep.td).mean().ffill().bfill()

        elif self.load_profile in ["const", "constant"]:
            self.flows_apriori["demand"] = self.consumption_yrl / (365 * 24)

        elif isinstance(self.load_profile, str):  # load_profile is a file name
            load_profile_file = self.scenario.paths.input / utils.set_extension(
                filename=self.load_profile, default_extension=".csv"
            )
            try:
                data = utils.read_timeseries(
                    path_input_file=load_profile_file,
                    timezone=self.scenario.location.timezone,
                    resampling_dti=self.scenario.times.sim.dti,
                )
            except IndexError as exc:
                raise IndexError(f"Failed to read load profile for block {self.name}: {exc}")

            if data.shape[1] != 1:
                self.scenario.logger.warning(
                    f"Input file {load_profile_file} for parameter "
                    f'"load_profile" in block "{self.name}" has more than one column. '
                    f"Sum of all columns is calculated for load profile."
                )

            data = data.sum(axis=1)[self.flows_apriori.index]  # convert to series and slice to sim timeframe
            self.flows_apriori["demand"] = data
        else:
            raise ValueError(f'Parameter "load_profile" in block "{self.block.name}" is not valid')

        if not self.scenario.settings.largescalemode:
            # feather requires a DataFrame with a default index, so frame the Series and move the index into a column
            self.flows_apriori["demand"].to_frame().reset_index().to_feather(
                self.scenario.paths.create_result_path(suffix=f"{self.scenario.name}_{self.name}_flow.feather")
            )


class ControllableSource(SourceBlock):
    _SIZE_NAMES = [("block", "kW")]
    _FLOW_NAMES = ["out"]

    def init_pois(self):
        super().init_pois()
        self.pois["block"] = eco.POI.create(
            name="block",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex=eco.CapexParams(
                spec=self.capex_spec,
                consider_preexisting=self.capex_block_preexisting,
                ls=self.ls,
                ccr=self.ccr,
            ),
            mntex=eco.MntexParams(spec=self.mntex_spec),
            opex=eco.OpexParams(spec_energy=self.opex_spec),
            name_size="block",
            name_flow="out",
        )

    def __init__(self, name: str, scenario: scn.Scenario, **kwargs):
        super().__init__(
            name=name,
            scenario=scenario,
            params=None,
            flow_apriori_names=None,
            parent=scenario,
            **kwargs,
        )


class GridConnection(ElectricBlock):
    _SIZE_NAMES = [("g2s", "kW"), ("s2g", "kW")]
    _FLOW_NAMES = ["in", "out"]

    def init_pois(self):
        super().init_pois()
        self.pois["g2s"] = eco.POI.create(
            name="g2s",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex=eco.CapexParams(
                spec=self.capex_spec,
                consider_preexisting=self.capex_g2s_preexisting,
                ls=self.ls,
                ccr=self.ccr,
            ),
            mntex=eco.MntexParams(spec=self.mntex_spec),
            name_size="g2s",
            name_flow="out",
        )

        self.pois["s2g"] = eco.POI.create(
            name="s2g",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex=eco.CapexParams(
                spec=self.capex_spec,
                consider_preexisting=self.capex_s2g_preexisting,
                ls=self.ls,
                ccr=self.ccr,
            ),
            mntex=eco.MntexParams(spec=self.mntex_spec),
            name_size="s2g",
            name_flow="in",
        )

        self.power_circles.append(("in", "out"))

    def __init__(self, name: str, scenario: scn.Scenario, **kwargs):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=None,
            parent=scenario,
            **kwargs,
        )

        self.peak_storages = dict()

        peak_period = peak_periods.PeakPowerPeriodFreq[self.peak_period.upper()]
        peak_period_start = peak_periods.PeakPowerPeriodStart[self.peak_period_start.upper()]

        self.peak_period_measurement = time.Timestep.from_str(self.peak_period_measurement)

        timeframe = self.scenario.times.sim

        self.peak_periods, self.peak_periods_activation = peak_periods.get_peak_periods(
            timeframe=timeframe,
            peak_period=peak_period,
            peak_period_start=peak_period_start,
            peak_power_init=self.peak_power_init,
        )

        steps_per_period = int(self.peak_period_measurement.hours / timeframe.timestep.hours)
        len_timeframe = len(timeframe.dti)
        t = np.arange(len_timeframe)
        t_extd = np.arange(len_timeframe + 1)

        # SOC of virtual storage has to be 0 at the start of each new measurement period
        self.peak_periods_soc_limit = pd.Series(
            data=(t_extd % steps_per_period != 0).astype(int),
            index=timeframe.dti_extd,
        )

        # outflow out of virtual storage for peak shaving is only allowed in the last timestep of a measurement period
        # activate outflow for all periods, if there are not at least 2 timesteps per measurement period
        self.peak_periods_storage_flush = pd.Series(
            data=((t % steps_per_period == steps_per_period - 1).astype(int)) if steps_per_period >= 2 else 1.0,
            index=timeframe.dti,
        )

        peak_period_pois = {
            period.label: eco.POI.create(
                name=period.label,
                eco=self.scenario.eco_params,
                data_dir=self.scenario.paths.input,
                opex=eco.OpexParams(
                    spec_peak=self.opex_spec_peak,
                    frac_peak=period.fraction,
                ),
            )
            for period in self.peak_periods.itertuples(index=False)
        }

        self.pois.update(peak_period_pois)
        for poi in peak_period_pois.values():
            self.aggregator.add_block(poi)

        self.flows[list(self.peak_periods.keys())] = 0.0

        if not self.markets:
            raise ValueError(
                f'Block "{self.name}": No markets defined! '
                f"At least one market has to be defined to buy and sell energy to the grid."
            )

        self.subblocks = {
            market: GridMarket(name=market, scenario=self.scenario, params=None, parent=self) for market in self.markets
        }
        del self.markets

    def params_preprocessing(self):
        self.expansion_equal = True if self.invest_g2s == "equal" or self.invest_s2g == "equal" else False

        self.init_equalizable_variables(name_vars=["invest_s2g", "invest_g2s"])
        self.init_equalizable_variables(name_vars=["size_g2s_preexisting", "size_s2g_preexisting"])
        self.init_equalizable_variables(name_vars=["size_g2s_max", "size_s2g_max"])

    def _build_poi_evaluation_kwargs(self, poi: eco.POI, **kwargs) -> dict[str, Any]:
        kwargs_eval = super()._build_poi_evaluation_kwargs(poi, **kwargs)

        if poi.name not in self.peak_periods.index:
            return kwargs_eval

        row = self.peak_periods.loc[poi.name]

        kwargs_eval["power_peak"] = row["peak_power"]
        return kwargs_eval

    def calc_results_energies(self):
        super().calc_results_energies()
        self.scenario.energies["sources"].add_energy(self.energies["out"])
        self.scenario.energies["sinks"].add_energy(self.energies["in"])


class GridMarket(ElectricBlock):
    _SIZE_NAMES = [("g2s", "kW"), ("s2g", "kW")]
    _FLOW_NAMES = ["in", "out"]

    def init_pois(self):
        super().init_pois()
        self.pois["g2s"] = eco.POI.create(
            name="g2s",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex=eco.OpexParams(spec_energy=self.opex_spec_g2s),
            name_size="g2s",
            name_flow="out",
        )

        self.pois["s2g"] = eco.POI.create(
            name="s2g",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex=eco.OpexParams(spec_energy=self.opex_spec_s2g),
            name_size="s2g",
            name_flow="in",
        )

    def __init__(self, name: str, scenario: scn.Scenario, params, parent, **kwargs):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=params,
            parent=parent,
            **kwargs,
        )


class StorageBlock(ElectricBlock, ABC):
    _SIZE_NAMES = [("storage", "kWh")]
    _FLOW_NAMES = ["in", "out", "bat_in", "bat_out"]
    _STATE_NAMES = [
        "energy",
        "soc",
        "soh",
        "q_loss_cal",
        "q_loss_cyc",
        "soc_min",
        "soc_max",
    ]

    def init_pois(self):
        super().init_pois()
        self.pois["storage"] = eco.POI.create(
            name="storage",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex=eco.CapexParams(
                spec=self.capex_spec,
                consider_preexisting=self.capex_storage_preexisting,
                ls=self.ls,
                ccr=self.ccr,
            ),
            mntex=eco.MntexParams(spec=self.mntex_spec),
            name_size="storage",
        )

        self.pois["in"] = eco.POI.create(
            name="in",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex=eco.OpexParams(spec_energy=self.opex_spec),
            name_flow="in",
        )

        self.pois["out"] = eco.POI.create(
            name="out",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            name_flow="out",
        )

        self.pois["bat_in"] = eco.POI.create(
            name="bat_in",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            name_flow="bat_in",
        )

        self.pois["bat_out"] = eco.POI.create(
            name="bat_out",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            name_flow="bat_out",
        )

        self.power_circles.append(("in", "out"))
        self.power_circles.append(("bat_in", "bat_out"))

    def params_preprocessing(self):
        super().params_preprocessing()
        # soc_min and soc_max are optional -> unset limits mean the full SOC window is usable
        for name_param, default in [("soc_min", 0.0), ("soc_max", 1.0)]:
            if getattr(self, name_param, None) is None:
                setattr(self, name_param, default)

        if self.soc_min >= self.soc_max:
            raise ValueError(f'"{self.name}": soc_min ({self.soc_min}) must be smaller than soc_max ({self.soc_max})')

    def update_soc_limits(self, ts: pd.Timestamp):
        """
        Write the SOC limits valid from timestamp ts onwards into the block's states.

        Capacity fade is modeled as a symmetric shrinking of the usable SOC window around SOC 0.5, so the limits
        depend on the block's SOH at ts. The user defined limits are applied on top: on either side, the more
        restrictive of the two limits is used.
        """
        soc_min_aging = (1 - self.states.loc[ts, "soh"]) / 2
        soc_max_aging = 1 - soc_min_aging

        soc_min = max(soc_min_aging, self.soc_min)
        soc_max = min(soc_max_aging, self.soc_max)

        if soc_min >= soc_max:
            # user defined and aging window do not overlap -> no SOC satisfies both
            self.scenario.logger.warning(
                f'"{self.name}": user defined SOC window [{self.soc_min}, {self.soc_max}] does not overlap the '
                f"window [{soc_min_aging:.3f}, {soc_max_aging:.3f}] left by aging at a SOH of "
                f"{self.states.loc[ts, 'soh']:.3f} - ignoring the user defined limits"
            )
            soc_min, soc_max = soc_min_aging, soc_max_aging

        self.states.loc[ts:, "soc_min"] = soc_min
        self.states.loc[ts:, "soc_max"] = soc_max

    def __init__(
        self,
        name: str,
        scenario: scn.Scenario,
        flow_apriori_names: list = None,
        params: dict = None,
        parent: BaseBlock | scn.Scenario = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=flow_apriori_names,
            params=params,
            parent=parent,
            **kwargs,
        )

        def calc_loss_rate_per_period(
            period: pd.Timedelta = pd.Timedelta(hours=1),
        ) -> float:
            """
            convert self-discharge rate (sdr) per month of a battery storage to a loss rate (lr) per target time step.
            oemof specifies one hour as the target time step for the loss rate.
            """
            ratio_timestep = period / pd.Timedelta("30 days")  # assumption: 30 days per month
            return 1 - (1 - self.sdr) ** ratio_timestep

        self.loss_rate_per_hour = calc_loss_rate_per_period(period=pd.Timedelta(hours=1))
        self.loss_rate_per_ts = calc_loss_rate_per_period(period=self.scenario.timestep.td)
        delattr(self, "sdr")

        # set initial SOC
        self.states.loc[self.scenario.times.eval.start, "soc"] = self.soc_init
        delattr(self, "soc_init")

        # set initial SOH
        self.states.loc[self.scenario.times.eval.start, "soh"] = 1 - self.q_loss_cal_init - self.q_loss_cyc_init

        # set initial calendric loss
        self.states.loc[self.scenario.times.eval.start, "q_loss_cal"] = self.q_loss_cal_init
        delattr(self, "q_loss_cal_init")

        # set inital cyclic loss
        self.states.loc[self.scenario.times.eval.start, "q_loss_cyc"] = self.q_loss_cyc_init
        delattr(self, "q_loss_cyc_init")

        self.update_soc_limits(ts=self.scenario.times.eval.start)

        # initialization of aging model after all blocks are initialized to get temp from pv blocks
        self.aging_model = None

    def pre_scenario(self, **kwargs):
        super().pre_scenario(**kwargs)
        self.aging_model = bat.BatteryPackModel.from_block(self)

    def calc_results_flows(self):
        super().calc_results_flows()
        # ToDo: what is bat_total required for?
        self.flows["bat_total"] = self.flows.get(key="bat_out", default=0) - self.flows.get(key="bat_in", default=0)


class StationaryBattery(StorageBlock):
    def __init__(
        self,
        name: str,
        scenario: scn.Scenario,
        **kwargs,
    ):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=None,
            parent=scenario,
            **kwargs,
        )

    def initialize_efficiencies(self):
        self.eff["chg"] = self.eff_acdc if self.system == "ac" else 1
        self.eff["dis"] = self.eff_dcac if self.system == "ac" else 1

        # necessary for common efficiency definition with ElectricFleetUnit
        self.eff["chg_int"] = self.eff["chg"]
        self.eff["dis_int"] = self.eff["dis"]

        for attr in ["eff_acdc", "eff_dcac"]:
            delattr(self, attr)

        super().initialize_efficiencies()


class Fleet(SinkBlock):
    _FLOW_NAMES = ["in", "out"]

    def params_preprocessing(self):
        super().params_preprocessing()
        # pwr_lim_f2s may be set to 'equal' to reuse pwr_lim_s2f
        if self.pwr_lim_f2s == "equal":
            self.pwr_lim_f2s = self.pwr_lim_s2f

    def init_pois(self):
        super().init_pois()
        self.pois["f2s"] = eco.POI.create(
            name="f2s",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex=eco.OpexParams(spec_energy=self.opex_spec_f2s),
            name_flow="out",
        )

        self.pois["s2f"] = eco.POI.create(
            name="s2f",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex=eco.OpexParams(spec_energy=self.opex_spec_s2f),
            name_flow="in",
        )

    def __init__(self, name: str, scenario: scn.Scenario, **kwargs):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=None,
            parent=scenario,
            **kwargs,
        )

        self.demand = None
        self.log = None

        types_units = {self.scenario.parameters[(subfleet, "type_unit")] for subfleet in self.subfleets}

        if types_units <= {"ev", "icev"}:
            self.is_vehicle_fleet = True
            cls_demand = mobility.VehicleFleetDemand
        elif types_units <= {"mb"}:
            self.is_vehicle_fleet = False
            cls_demand = mobility.BatteryFleetDemand
        else:
            raise ValueError(
                f"Fleet {self.name} has (a) both vehicle and battery subfleets or (b) invalid unit types assigned."
            )

        # region create and fill demand object
        if self.data_source in ["usecases", "demand"]:
            self.demand = cls_demand(dti=self.scenario.times.sim.dti)
            self.scenario.block_registry.setdefault("DispatchFleet", {})[self.name] = self

        if self.data_source == "usecases":
            path_demand = (
                self.scenario.paths.create_result_path(suffix=f"{self.scenario.name}_{self.name}_demand.feather")
                if not self.scenario.settings.largescalemode
                else None
            )
            self.demand.from_usecases(
                path_usecases=self.scenario.paths.input
                / utils.set_extension(filename=self.filename, default_extension=".csv"),
                path_timeframe_mapper=self.scenario.paths.input / f"{self.filename_mapper}.py",
                path_demand=path_demand,
                key_timeframe_mapper=self.name,
                subfleets=self.subfleets,
            )

        elif self.data_source == "demand":
            self.demand.from_file(
                path_demand=(
                    self.scenario.paths.input / utils.set_extension(filename=self.filename, default_extension=".csv")
                ),
                dti=self.scenario.times.sim.dti,
            )

        elif self.data_source in ["log", "logfile"]:
            self.log = self.read_logfile()

        else:
            raise ValueError(f'Block "{self.name}": invalid data source')
        # endregion

        # create subfleets
        [SubFleet(name=item, scenario=self.scenario, parent=self) for item in self.subfleets]

        # check for dispatch inconsistency after subblocks are available
        if any(
            (subfleet.rex is not None and subfleet.parent.data_source not in ["usecases", "demand"])
            for subfleet in self.subblocks.values()
        ):
            raise ValueError("all subfleets with range extension must be actively dispatched")

    def read_logfile(self) -> pd.DataFrame:
        """
        Read in a predetermined log file for group behavior.
        """

        try:
            df = utils.read_timeseries(
                path_input_file=(
                    self.scenario.paths.input / utils.set_extension(filename=self.filename, default_extension=".csv")
                ),
                timezone=self.scenario.location.timezone,
                multiheader=True,
            )  # Normal resampling cannot be used as consumption must be
        # meaned, while booleans, distances and dsocs must not.
        except IndexError as exc:
            raise IndexError(f"Failed to load input log for block {self.name}: {exc}")

        # Timedelta of frequency of log file
        freq_log = pd.infer_freq(df.index).lower()
        # pd.Timedelta('h') fails --> add '1' --> pd.Timedelta('1h')
        freq_log = pd.Timedelta((freq_log if freq_log[0].isdigit() else "1" + freq_log))

        # Compare Timedelta objects instead of strings to avoid problems (1h vs. 60min)
        if freq_log != self.scenario.timestep.td:
            self.scenario.logger.warning(
                f'Block "{self.name}": log file does not match specified timestep - Resampling'
            )

            cols = df.columns  # save original column sorting to apply after resampling
            cols_consumption = df.columns[df.columns.get_level_values(1) == "consumption"]
            cols_dist = df.columns[df.columns.get_level_values(1) == "dist"]
            cols_bool = df.columns.difference(cols_consumption).difference(cols_dist)
            # mean ensures equal energy consumption after downsampling, ffill and bfill fill upsampled NaN values
            df_new = pd.DataFrame()
            df_new[cols_consumption] = df[cols_consumption].resample(self.scenario.timestep.td).mean().ffill().bfill()
            df_new[cols_dist] = df[cols_dist].resample(self.scenario.timestep.td).sum().ffill().bfill()
            df_new[cols_bool] = df[cols_bool].resample(self.scenario.timestep.td).ffill().bfill()
            df = df_new[cols]  # ensure right sorting

        if not (self.scenario.times.sim.dti.isin(df.index).all()):
            self.scenario.logger.error(
                f'Block "{self.name}": Input timeseries data does not cover simulation timeframe'
            )

        return df.loc[self.scenario.times.sim.dti]  # need dsoc for last timestep


class SubFleet(NonElectricBlock):
    def __init__(self, name: str, scenario: scn.Scenario, parent, **kwargs):
        # subfleet parameters contain FleetUnit parameters -> split parameters for FleetUnits and SubFleet
        params = scenario.parameters.loc[name]
        params_subfleet = {key: params.pop(key) if key in params else None for key in ["num", "type_unit", "rex"]}

        super().__init__(name=name, scenario=scenario, params=params_subfleet, parent=parent, **kwargs)

        self.demand = None
        self.log = None

        cls_fu = {
            "ev": ElectricVehicle,
            "icev": CombustionVehicle,
            "mb": MobileBattery,
        }.get(self.type_unit)
        self.unit_names = [f"{self.name}{i}" for i in range(self.num)]
        [cls_fu(name=name, scenario=self.scenario, params=params, parent=self) for name in self.unit_names]

        if params.get("mode_scheduling") in scenario.apriori_lvls:  # mode scheduling attr is in FleetUnit
            self.scenario.block_registry.setdefault("SubFleetScheduling", {})[self.name] = self

        if getattr(self, "invest", False) and self.data_source in ["usecases", "demand"]:
            self.scenario.logger.Error(
                f'Subfleet "{self.name}": investment not implemented for data source "{self.data_source}"'
            )

    def pre_scenario(self, **kwargs):
        self.log = self.parent.log.loc[:, self.parent.log.columns.get_level_values(0).str.contains(self.name)]
        super().pre_scenario(**kwargs)


class FleetUnit(BaseBlock):
    # energy consumed by the customer while the unit is rented out. This is not an optimization flow, but is derived from the dispatch log in pre_scenario (see below).
    _FLOW_NAMES = ["consumption"]

    def init_pois(self):
        self.pois["glider"] = eco.POI.create(
            name="glider",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex=eco.CapexParams(
                fix=self.capex_fix_glider,
                ls=self.ls,
                ccr=self.ccr,
                consider_preexisting=self.capex_glider_preexisting,
            ),
            mntex=eco.MntexParams(fix=self.mntex_fix_glider),
            opex=eco.OpexParams(spec_dist=self.opex_spec_dist),
            crev=eco.CrevParams(
                spec_dist=self.crev_spec_dist if hasattr(self, "crev_spec_dist") else 0.0,
                spec_time=self.crev_spec_time if hasattr(self, "crev_spec_time") else 0.0,
                spec_energy=self.crev_spec_energy if hasattr(self, "crev_spec_energy") else 0.0,
            ),
            name_flow="consumption",
        )

    def __init__(
        self,
        name: str,
        scenario: scn.Scenario,
        params: dict = None,
        parent: BaseBlock | scn.Scenario = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            scenario=scenario,
            params=params,
            parent=parent,
            **kwargs,
        )

        self.log = None

    def pre_scenario(self, **kwargs):
        """
        slice log file from subfleet
        """
        self.log = self.parent.log.loc[:, (self.name, slice(None))].droplevel(0, axis=1)

        # ensure that all columns are present in the log file
        for col_name, col_value in [
            ("atac", False),
            ("atdc", False),
            ("atbase", True),
            ("consumption", 0.0),
            ("dist", 0.0),
            ("dsoc", 0.0),
        ]:
            if col_name not in self.log.columns:
                self.log[col_name] = col_value

        # the consumption flow is not part of the optimization and is therefore filled from the dispatch log instead of being written back by the simulation result visitor
        self.flows["consumption"] = self.log["consumption"].astype(float)

        super().pre_scenario(**kwargs)

    def post_scenario(self):
        self.utilization = 1 - self.log["atbase"].mean()
        self.dist_eval = self.log["dist"].sum() if "dist" in self.log.columns else 0
        super().post_scenario()

    def _build_poi_evaluation_kwargs(self, poi: eco.POI, **kwargs) -> dict[str, Any]:
        kwargs_eval = super()._build_poi_evaluation_kwargs(poi, **kwargs)

        kwargs_eval["dist"] = self.log["dist"]
        # convert to active time -> vehicle not at base
        kwargs_eval["time"] = ~(self.log["atbase"].astype(bool))
        return kwargs_eval


class ElectricFleetUnit(StorageBlock, FleetUnit):
    _FLOW_NAMES = ["ext_ac", "ext_dc"]

    def init_pois(self):
        super().init_pois()

        self.pois["charger"] = eco.POI.create(
            name="charger",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex=eco.CapexParams(
                fix=self.capex_fix_charger,
                ls=self.ls,
                ccr=self.ccr,
                consider_preexisting=self.capex_charger_preexisting,
            ),
        )

        self.pois["ext_ac"] = eco.POI.create(
            name="ext_ac",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex=eco.OpexParams(spec_energy=self.opex_spec_ext_ac),
            name_flow="ext_ac",
        )

        self.pois["ext_dc"] = eco.POI.create(
            name="ext_dc",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex=eco.OpexParams(spec_energy=self.opex_spec_ext_dc),
            name_flow="ext_dc",
        )

    def __init__(self, name: str, scenario: scn.Scenario, parent: SubFleet, params: dict, **kwargs):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=[
                "p_int_chg",
                "p_ext_ac_chg",
                "p_ext_dc_chg",
                "p_int_dis",
                "p_ext_ac_dis",
                "p_ext_dc_dis",
            ],
            params=params,
            parent=parent,
            **kwargs,
        )

        self.apriori = True if self.mode_scheduling in self.scenario.apriori_lvls else False

        if any([size.invest for size in self.sizes.values()]) and self.mode_scheduling in self.scenario.apriori_lvls:
            raise ValueError(
                f'ElectricFleetUnit "{self.name}": size optimization not '
                f"implemented for a priori integration levels: {self.scenario.apriori_lvls}"
            )

    def params_preprocessing(self):
        super().params_preprocessing()
        # pwr_dis_max may be set to 'equal' to reuse pwr_chg_max
        if self.pwr_dis_max == "equal":
            self.pwr_dis_max = self.pwr_chg_max

    def initialize_efficiencies(self):
        self.eff["chg_int"] = {"ac": self.eff_chg_ac, "dc": self.eff_chg_dc}[self.parent.parent.system]
        self.eff["dis_int"] = {"ac": self.eff_dis_ac, "dc": self.eff_dis_dc}[self.parent.parent.system]
        super().initialize_efficiencies()


class CombustionVehicle(NonElectricBlock, FleetUnit):
    def __init__(self, name: str, scenario: scn.Scenario, parent: SubFleet, params: dict, **kwargs):
        super().__init__(
            name=name,
            scenario=scenario,
            params=params,
            parent=parent,
            **kwargs,
        )

        # delete parameters not needed for CombustionVehicles
        # ToDo: specify required parameters instead of obsolete ones
        for param in [
            "aging",
            "chemistry",
            "temp_battery",
            "q_loss_cal_init",
            "q_loss_cyc_init",
            "soc_init",
            "soc_target",
            "soc_return",
            "dsoc_buffer",
            "pwr_chg_max",
            "pwr_dis_max",
            "pwr_ext_ac_max",
            "pwr_ext_dc_max",
            "eff_storage_roundtrip",
            "eff_chg_ac",
            "eff_chg_dc",
            "eff_dis_ac",
            "eff_dis_dc",
            "sdr",
        ]:
            if hasattr(self, param):
                delattr(self, param)


class ElectricVehicle(ElectricFleetUnit):
    pass


class MobileBattery(ElectricFleetUnit):
    def __init__(self, name: str, scenario: scn.Scenario, parent: SubFleet, params: dict, **kwargs):
        # initialize for scenario files without these parameters
        self.opex_spec_dist = 0.0
        self.opex_spec_time = 0.0
        super().__init__(name=name, scenario=scenario, parent=parent, params=params, **kwargs)
