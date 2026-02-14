#!/usr/bin/env python3

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import oemof.solph as solph
import pandas as pd
import windpowerlib
from typing_extensions import override

from revoletion import battery as bat
from revoletion import data_manager, mobility, utils
from revoletion import eco


if TYPE_CHECKING:
    import datetime
    from revoletion import simulation


class BlockScenarioInterface(ABC):
    @abstractmethod
    def pre_scenario(self) -> None:
        """
        Trigger actions to be executed after all inits.
        """
        ...

    @abstractmethod
    def pre_horizon(self, horizon: simulation.PredictionHorizon) -> None:
        """
        Trigger actions to be executed before each horizon.
        """
        ...

    @abstractmethod
    def post_horizon(self, horizon: simulation.PredictionHorizon) -> None:
        """
        Trigger actions to be executed after each horizon.
        """
        ...

    @abstractmethod
    def post_scenario(self) -> None:
        """
        Trigger actions to be executed after the scenario has been run.
        """
        ...


class BaseBlock(BlockScenarioInterface, ABC):
    """
    abstract class
    """

    def init_pois(self):
        # add a new POI to block.pois
        pass

    def init_states(self):
        # add a new column to block.states
        pass

    def __init__(
        self,
        name: str,
        scenario: simulation.Scenario,
        params: dict = None,
        parent: BaseBlock | simulation.Scenario = None,
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

        self.aggregator = eco.Aggregator.create(name=self.name)

        self.states = pd.DataFrame(index=self.scenario.times.sim.dti_extd, dtype="float64")
        self.init_states()

        self.pois = dict()
        self.init_pois()
        for poi in self.pois.values():
            self.aggregator.add_block(poi)
        self.parent.aggregator.add_block(self.aggregator)

        # ToDo: find solution for size unit
        self.sizes = {
            poi.name_size: eco.size.Size.create_from_block(name=poi.name_size, block=self, unit="kW")
            for poi in self.pois.values()
            if poi.name_size is not None
        }

        self.flows = pd.DataFrame(
            index=self.scenario.times.sim.dti,
            columns=["total"] + [poi.name_flow for poi in self.pois.values() if poi.name_flow is not None],
            data=0.0,
            dtype=float,
        )

        self.energies = {
            flow: eco.EnergyEvaluator(name=flow, eco=self.scenario.eco_params) for flow in self.flows.columns
        }

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
    def pre_scenario(self):
        """
        trigger actions to be executed after all inits
        """
        for subblock in self.subblocks.values():
            subblock.pre_scenario()

    @override
    def pre_horizon(self, horizon: simulation.PredictionHorizon):
        for subblock in self.subblocks.values():
            subblock.pre_horizon(horizon=horizon)

    @override
    def post_horizon(self, horizon: simulation.PredictionHorizon):
        for subblock in self.subblocks.values():
            subblock.post_horizon(horizon=horizon)

    @override
    def post_scenario(self):
        for subblock in self.subblocks.values():
            subblock.post_scenario()

        # calculate results
        self.calc_results_economics()

    def calc_results_economics(self):
        # ToDo: remove this part
        # calculate economic results
        for poi in self.pois.values():
            poi.calc_results(
                sizes=self.sizes,
                flows=self.flows,
            )


class NonElectricBlock(BaseBlock): ...


class ElectricBlock(BaseBlock, ABC):
    def __init__(
        self,
        name: str,
        scenario: simulation.Scenario,
        flow_apriori_names: list = None,
        params: dict = None,
        parent: BaseBlock | simulation.Scenario = None,
    ):
        self.power_circles = []

        super().__init__(name=name, scenario=scenario, params=params, parent=parent)

        # empty list not possible as default argument as it is mutable
        flow_apriori_names = flow_apriori_names if flow_apriori_names is not None else []

        self.components = dict()
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

    def pre_horizon(self, horizon: simulation.PredictionHorizon):
        self.define_oemof_components(horizon=horizon)
        horizon.es.add(*self.components.values())

        super().pre_horizon(horizon=horizon)  # executes pre_horizon for subblocks

    def post_horizon(self, horizon: simulation.PredictionHorizon):
        super().post_horizon(horizon=horizon)  # executes post_horizon for subblocks

        self.get_horizon_results(horizon=horizon)

    def post_scenario(self):
        self.calc_results_flows()
        self.calc_results_energies()

        super().post_scenario()

    @abstractmethod
    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None): ...

    @abstractmethod
    def get_horizon_results(self, horizon: simulation.PredictionHorizon): ...

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
        for energy in self.energies.values():
            energy.calc_results(self.flows.loc[self.scenario.times.eval.dti, energy.name])


class SourceBlock(ElectricBlock, ABC):
    @abstractmethod
    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None): ...

    @abstractmethod
    def get_horizon_results(self, horizon: simulation.PredictionHorizon): ...

    def calc_results_energies(self):
        super().calc_results_energies()
        # self.scenario.energies.loc[("sources", "pro"), :] += self.energies.loc["total", :]
        # ToDo: fix this
        self.scenario.energies["sources"].add_energy(self.energies["total"])


class SinkBlock(ElectricBlock, ABC):
    @abstractmethod
    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None): ...

    @abstractmethod
    def get_horizon_results(self, horizon: simulation.PredictionHorizon): ...

    def calc_results_energies(self):
        super().calc_results_energies()
        # self.scenario.energies.loc[("sinks", "del"), :] -= self.energies.loc["total", :]
        # ToDo: fix this
        self.scenario.energies["sinks"].add_energy(self.energies["total"])


class SystemCore(ElectricBlock):
    def init_pois(self):
        super().init_pois()

        self.pois["acdc"] = eco.Evaluator.create(
            name="acdc",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex_spec=self.capex_spec,
            capex_ccr=self.ccr,
            consider_preexisting=self.capex_preexisting_acdc,
            ls=self.ls,
            mntex_spec=self.mntex_spec,
            opex_spec=self.opex_spec,
        )

        self.pois["dcac"] = eco.Evaluator.create(
            name="dcac",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex_spec=self.capex_spec,
            capex_ccr=self.ccr,
            consider_preexisting=self.capex_preexisting_dcac,
            ls=self.ls,
            mntex_spec=self.mntex_spec,
            opex_spec=self.opex_spec,
        )

        self.power_circles.append(("acdc", "dcac"))

    def __init__(self, name: str, scenario):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=None,
            parent=scenario,
        )

    def params_preprocessing(self):
        self.expansion_equal = True if self.invest_acdc == "equal" or self.invest_dcac == "equal" else False

        self.init_equalizable_variables(name_vars=["invest_acdc", "invest_dcac"])
        self.init_equalizable_variables(name_vars=["size_preexisting_acdc", "size_preexisting_dcac"])
        self.init_equalizable_variables(name_vars=["size_max_acdc", "size_max_dcac"])

    @override
    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None):
        """
        pre horizon method
        x denotes the flow measurement point in results

          dc          ac
          |-x--dcac-->|
          |           |
          |<---acdc-x-|
        """

        self.components["ac"] = solph.Bus()
        self.components["dc"] = solph.Bus()

        self.components["acdc"] = solph.components.Converter(
            inputs={
                self.components["ac"]: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=self.pois["acdc"].spec_ep_invest,
                        existing=self.sizes["acdc"].preexisting,
                        maximum=self.sizes["acdc"].expansion_max,
                    ),
                    variable_costs=self.pois["acdc"].spec_ep_operation[horizon.ph.dti],
                )
            },
            outputs={self.components["dc"]: solph.Flow(variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.components["dc"]: self.eff["acdc"]},
        )

        self.components["dcac"] = solph.components.Converter(
            inputs={
                self.components["dc"]: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=self.pois["dcac"].spec_ep_invest,
                        existing=self.sizes["dcac"].preexisting,
                        maximum=self.sizes["dcac"].expansion_max,
                    ),
                    variable_costs=self.pois["dcac"].spec_ep_operation[horizon.ph.dti],
                )
            },
            outputs={self.components["ac"]: solph.Flow(variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.components["ac"]: self.eff["dcac"]},
        )

        horizon.constraints.add_invest_costs(
            invest=(self.components["ac"], self.components["acdc"]),
            capex_spec=self.pois["acdc"].capex.spec,
            invest_type="flow",
        )

        horizon.constraints.add_invest_costs(
            invest=(self.components["dc"], self.components["dcac"]),
            capex_spec=self.pois["dcac"].capex.spec,
            invest_type="flow",
        )

        if self.expansion_equal:
            # add a tuple of tuples to the list of equal variables of the scenario
            horizon.constraints.add_equal_invests(
                [
                    {"in": self.components["dc"], "out": self.components["dcac"]},
                    {"in": self.components["ac"], "out": self.components["acdc"]},
                ]
            )

    def get_horizon_results(self, horizon: simulation.PredictionHorizon):
        """
        post horizon method
        """
        self.sizes["acdc"].expansion = horizon.results[(self.components["ac"], self.components["acdc"])]["scalars"][
            "invest"
        ]
        self.sizes["dcac"].expansion = horizon.results[(self.components["dc"], self.components["dcac"])]["scalars"][
            "invest"
        ]

        self.flows.loc[horizon.ch.dti, "acdc"] = horizon.results[(self.components["ac"], self.components["acdc"])][
            "sequences"
        ]["flow"][horizon.ch.dti]
        self.flows.loc[horizon.ch.dti, "dcac"] = horizon.results[(self.components["dc"], self.components["dcac"])][
            "sequences"
        ]["flow"][horizon.ch.dti]

    def calc_results_flows(self):
        """
        post scenario method
        """
        super().calc_results_flows()


class RenewableSource(SourceBlock, ABC):
    """
    abstract class
    """

    def init_pois(self):
        super().init_pois()
        self.pois["block"] = eco.Evaluator.create(
            name="block",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex_spec=self.capex_spec,
            capex_ccr=self.ccr,
            consider_preexisting=self.capex_preexisting_block,
            ls=self.ls,
            mntex_spec=self.mntex_spec,
            opex_spec=self.opex_spec,
            flow_name="out",
        )

        self.pois["curt"] = eco.Evaluator.create(
            name="curt",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            flow_name="curt",
        )

        self.pois["pot"] = eco.Evaluator.create(
            name="pot",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            flow_name="pot",
        )

    def __init__(self, name: str, scenario: "simulation.Scenario"):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=None,
            parent=scenario,
        )
        self.data = None  # todo move to a priori flows (except for wind speed and ambient temp)
        self.get_ts_data()

        self.share_curtailment = None

    @abstractmethod
    def get_ts_data(self): ...

    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected      name_bus
          |                   |
          |<--x----name_out---|<--name_src
          |                   |
          |                   |-->name_exc
        """

        self.bus_connected = self.scenario.block_registry.get("TopLevelBlock", {})["core"].components[self.system]

        self.components["bus"] = solph.Bus()

        self.components["outflow"] = solph.components.Converter(
            inputs={self.components["bus"]: solph.Flow()},
            outputs={self.bus_connected: solph.Flow()},
            conversion_factors={self.bus_connected: self.eff["block"]},
        )

        # Curtailment has to be disincentivized in the optimization to force optimizer to charge storage or commodities
        # instead of curtailment. 2x cost_eps is required as SystemCore also has ccost_eps in charging direction.
        # All other components such as converters and storages only have cost_eps in the output direction.
        self.components["exc"] = solph.components.Sink(inputs={self.components["bus"]: solph.Flow()})

        self.components["src"] = solph.components.Source(
            outputs={
                self.components["bus"]: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=self.pois["block"].spec_ep_invest,
                        existing=self.sizes["block"].preexisting,
                        maximum=self.sizes["block"].expansion_max,
                    ),
                    fix=self.data.loc[horizon.ph.dti, "power_spec"],
                    variable_costs=self.pois["block"].spec_ep_operation[horizon.ph.dti],
                )
            }
        )

        horizon.constraints.add_invest_costs(
            invest=(self.components["src"], self.components["bus"]),
            capex_spec=self.pois["block"].capex.spec,
            invest_type="flow",
        )

    def get_horizon_results(self, horizon: simulation.PredictionHorizon):
        """
        post horizon method
        """
        self.sizes["block"].expansion = horizon.results[(self.components["src"], self.components["bus"])]["scalars"][
            "invest"
        ]

        self.flows.loc[horizon.ch.dti, "out"] = horizon.results[(self.components["outflow"], self.bus_connected)][
            "sequences"
        ]["flow"][horizon.ch.dti]
        self.flows.loc[horizon.ch.dti, "pot"] = horizon.results[(self.components["src"], self.components["bus"])][
            "sequences"
        ]["flow"][horizon.ch.dti]
        self.flows.loc[horizon.ch.dti, "curt"] = horizon.results[(self.components["bus"], self.components["exc"])][
            "sequences"
        ]["flow"][horizon.ch.dti]

    def calc_results_energies(self):
        super().calc_results_energies()
        # add curt and pot to scenario.energies
        # self.scenario.energies.loc[("renewable", "act"), :] += self.energies.loc["out", :]
        #
        # # pandas creates a RuntimeWarning at division by 0 -> try/except does not work
        # if self.energies.loc["pot", "sim"] == 0:
        #     self.scenario.logger.warning(f"Block {self.name}: Curtailment share calculation: division by zero")
        # else:
        #     self.share_curtailment = self.energies.loc["curt", "sim"] / self.energies.loc["pot", "sim"]
        # ToDo: fix this
        self.scenario.energies["renewable_actual"].add_energy(self.energies["total"])
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
        except data_manager.DataProviderError as e:
            raise RuntimeError(f"Failed to retrieve timeseries data for block {self.name}") from e
        path_input_file = (
            self.scenario.paths.input / utils.set_extension(filename=self.filename, default_extension=".csv")
            if "file" in self.data_source
            else None
        )


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
                self.data = utils.read_timeseries_csv(
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
            self.data.to_csv(self.scenario.paths.create_result_path(suffix=f"{self.scenario.name}_{self.name}_log.csv"))


class FixedDemand(SinkBlock):
    _SLP_IDS = ["h0", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "l0", "l1", "l2", "h25", "g25", "l25", "s25", "p25"]

    def init_pois(self):
        super().init_pois()
        self.pois["block"] = eco.Evaluator.create(
            name="block",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            consider_preexisting=self.capex_preexisting_metering,
            capex_fix=self.capex_fix_metering,
            mntex_fix=self.mntex_fix_metering,
            crev_spec=self.crev_spec,
            flow_name="in",
        )

    def __init__(self, name: str, scenario):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=["demand"],
            params=None,
            parent=scenario,
        )

        self.get_flows_apriori()

    @staticmethod
    def get_slp(
        slp_id: str,
        ts_start: pd.Timestamp,
        ts_end: pd.Timestamp,
        holiday_dates: list[datetime.date] = None,
        path_data: Path = None,
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

        if path_data is None:
            # for standalone use: get revoletion data path
            import importlib.resources as pkg_resources

            try:
                import revoletion
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "revoletion not found. Please install revoletion or provide path_data argument."
                )
            path_data = pkg_resources.files(revoletion.data)

        # read SLP data, do not set index here, as we need to convert time column first
        data = pd.read_csv(
            path_data / "slp_bdew.csv",
            index_col=[],
        )

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
                path_data=self.scenario.paths.data_persist,
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
                data = utils.read_timeseries_csv(
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
            self.flows_apriori["demand"].to_csv(
                self.scenario.paths.create_result_path(suffix=f"{self.scenario.name}_{self.name}_flow.csv")
            )

    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected
          |
          |-x->name_snk
          |
        """

        self.bus_connected = self.scenario.block_registry.get("TopLevelBlock", {})["core"].components[self.system]

        self.components["snk"] = solph.components.Sink(
            inputs={
                self.bus_connected: solph.Flow(nominal_capacity=1, fix=self.flows_apriori["demand"][horizon.ph.dti])
            }
        )

    def get_horizon_results(self, horizon: simulation.PredictionHorizon):
        """
        post horizon method
        """
        self.flows.loc[horizon.ch.dti, "in"] = horizon.results[(self.bus_connected, self.components["snk"])][
            "sequences"
        ]["flow"][horizon.ch.dti]


class ControllableSource(SourceBlock):
    def init_pois(self):
        super().init_pois()
        self.pois["block"] = eco.Evaluator.create(
            name="block",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            ls=self.ls,
            capex_ccr=self.ccr,
            consider_preexisting=self.capex_preexisting_block,
            capex_spec=self.capex_spec,
            mntex_spec=self.mntex_spec,
            opex_spec=self.opex_spec,
            flow_name="out",
        )

    def __init__(self, name: str, scenario: simulation.Scenario):
        super().__init__(
            name=name,
            scenario=scenario,
            params=None,
            flow_apriori_names=None,
            parent=scenario,
        )

    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected
          |
          |<-name_gen
          |
        """

        self.bus_connected = self.scenario.block_registry.get("TopLevelBlock", {})["core"].components[self.system]

        self.components["src"] = solph.components.Source(
            outputs={
                self.bus_connected: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=self.pois["block"].spec_ep_invest,
                        existing=self.sizes["block"].preexisting,
                        maximum=self.sizes["block"].expansion_max,
                    ),
                    variable_costs=self.pois["block"].spec_ep_operation[horizon.ph.dti],
                )
            }
        )

        horizon.constraints.add_invest_costs(
            invest=(self.components["src"], self.bus_connected),
            capex_spec=self.pois["block"].capex.spec,
            invest_type="flow",
        )

    def get_horizon_results(self, horizon: simulation.PredictionHorizon):
        """
        post horizon method
        """
        self.sizes["block"].expansion = horizon.results[(self.components["src"], self.bus_connected)]["scalars"][
            "invest"
        ]

        self.flows.loc[horizon.ch.dti, "out"] = horizon.results[(self.components["src"], self.bus_connected)][
            "sequences"
        ]["flow"][horizon.ch.dti]


class GridConnection(ElectricBlock):
    def init_pois(self):
        super().init_pois()
        self.pois["g2s"] = eco.Evaluator.create(
            name="g2s",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            consider_preexisting=self.capex_preexisting_g2s,
            capex_spec=self.capex_spec,
            ls=self.ls,
            capex_ccr=self.ccr,
            mntex_spec=self.mntex_spec,
            flow_name="out",
        )

        self.pois["s2g"] = eco.Evaluator.create(
            name="s2g",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            consider_preexisting=self.capex_preexisting_s2g,
            capex_spec=self.capex_spec,
            ls=self.ls,
            capex_ccr=self.ccr,
            mntex_spec=self.mntex_spec,
            flow_name="in",
        )

        self.power_circles.append(("in", "out"))

    def __init__(self, name: str, scenario: simulation.Scenario):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=None,
            parent=scenario,
        )

        self.inflows = dict()
        self.outflows = dict()

        self.peak_periods = pd.DataFrame()
        self.bus_activation = pd.DataFrame()

        self.initialize_peak_tracking()

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
        self.init_equalizable_variables(name_vars=["size_preexisting_g2s", "size_preexisting_s2g"])
        self.init_equalizable_variables(name_vars=["size_max_g2s", "size_max_s2g"])

    def initialize_peak_tracking(self):
        # Create functions to extract relevant property of datetimeindex for peakshaving intervals
        periods_func = {
            "day": lambda x: x.strftime("%Y-%m-%d"),
            "week": lambda x: x.strftime("%Y-CW%W"),
            "month": lambda x: x.strftime("%Y-%m"),
            "quarter": lambda x: f"{x.year}-Q{(x.month - 1) // 3 + 1}",
            "year": lambda x: x.strftime("%Y"),
        }

        if self.peak_period not in periods_func.keys():
            raise ValueError(f'Block {self.name}: parameter "peak_period" must be one of {periods_func.keys()}')

        # Get dummies directly from the 'periods' data
        self.bus_activation = pd.get_dummies(
            self.scenario.times.sim.dti.to_series().map(periods_func[str(self.peak_period)])
        ).astype(int)

        # Create a series to store peak power values
        self.peak_periods = pd.DataFrame(
            index=self.bus_activation.columns,
            columns=["power"],
            data=self.peak_power_init,  # cumulative variable
            dtype="float64",
        )

        def process_period(period):
            dti_period = self.bus_activation[self.bus_activation[period] == 1].index
            dti_period_sim = dti_period[dti_period.isin(self.scenario.times.sim.dti)]  # remove non-sim timestamps

            # if interval is not part of dti_sim (happens for rh), dti is empty -> return 0
            if len(dti_period_sim) == 0:
                period_fraction = 0.0
            else:
                if period == "day":
                    start = dti_period_sim.min().normalize()
                    end = start + pd.DateOffset(days=1) - self.scenario.timestep.td
                elif period == "week":
                    start = dti_period_sim.min().normalize() - pd.Timedelta(days=dti_period_sim[0].weekday())
                    end = start + pd.DateOffset(weeks=1) - self.scenario.timestep.td
                elif period == "month":
                    start = dti_period_sim.min().normalize().replace(day=1)
                    end = start + pd.DateOffset(months=1) - self.scenario.timestep.td
                elif period == "quarter":
                    start = (
                        dti_period_sim.min()
                        .normalize()
                        .replace(day=1, month=((dti_period_sim[0].month - 1) // 3) * 3 + 1)
                    )
                    end = start + pd.DateOffset(months=3) - self.scenario.timestep.td
                elif period == "year":
                    start = dti_period_sim.min().normalize().replace(day=1, month=1)
                    end = start + pd.DateOffset(years=1) - self.scenario.timestep.td
                else:
                    start = dti_period_sim.min()
                    end = dti_period_sim.max()

                period_fraction = len(dti_period_sim) / len(pd.date_range(start, end, freq=self.scenario.timestep.td))

            return pd.Series(
                {
                    "period_fraction": period_fraction,
                    "start": dti_period.min(),
                    "end": dti_period.max(),
                }
            )

        # Apply the function to each period in peak_periods
        self.peak_periods[["period_fraction", "start", "end"]] = self.peak_periods.index.to_series().apply(
            process_period
        )

        self.n_peak_periods_yr = (
            (
                pd.date_range(
                    start=self.scenario.times.sim.start,
                    end=self.scenario.times.sim.start + pd.DateOffset(years=1),
                    freq=self.scenario.timestep.td,
                    inclusive="left",
                )
                .to_series()
                .apply(periods_func[str(self.peak_period)])
            )
            .unique()
            .size
        )

        # ToDo: implement PeakShaving in new eco structure
        # self.pois.update(
        #     {
        #         period: eco.EcoEvaluator(
        #             name=period,
        #             block=self,
        #             scenario=self.scenario,
        #             opex_config_peak=dict(spec_peak=self.opex_spec_peak),
        #         )
        #         for period in self.peak_periods.index
        #     }
        # )

    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected          name_bus
          |                        |
          |---name_inflow_1--x---->|
          |<--name_outflow_1--x----|
          |                        |---(GridMarket Instance)
          |---name_inflow_2--x---->|
          |<--name_outflow_2--x----|
          |                        |---(GridMarket Instance)

                     ...

          |---name_inflow_n--x---->|
          |<--name_outflow_n--x----|
        """

        self.bus_connected = self.scenario.block_registry.get("TopLevelBlock", {})["core"].components[self.system]

        self.components["bus"] = solph.Bus()

        self.inflows = {
            f"{self.name}_inflow_1": solph.components.Converter(
                # Peakshaving not implemented for feed-in into grid
                inputs={self.bus_connected: solph.Flow()},
                # Size optimization
                outputs={
                    self.components["bus"]: solph.Flow(
                        nominal_capacity=solph.Investment(
                            ep_costs=self.pois["s2g"].spec_ep_invest,
                            existing=self.sizes["s2g"].preexisting,
                            maximum=self.sizes["s2g"].expansion_max,
                        ),
                        variable_costs=self.scenario.cost_eps,
                    )
                },
                conversion_factors={self.components["bus"]: 1},
            )
        }

        self.components.update(self.inflows)

        self.outflows = {
            f"{self.name}_outflow_{period}": solph.components.Converter(
                # Size optimization: investment costs are assigned to first peakshaving interval only. The application of
                # constraints ensures that the optimized grid connection sizes of all peakshaving intervals are equal
                inputs={
                    self.components["bus"]: solph.Flow(
                        nominal_capacity=solph.Investment(
                            ep_costs=(self.pois["g2s"].spec_ep_invest if period == self.peak_periods.index[0] else 0),
                            existing=self.sizes["g2s"].preexisting,
                            maximum=self.sizes["g2s"].expansion_max,
                        )
                    )
                },
                # Peakshaving
                outputs={
                    self.bus_connected: solph.Flow(
                        nominal_capacity=(
                            solph.Investment(
                                ep_costs=(self.pois[period].spec_ep_peak if self.peakshaving else 0),
                                existing=self.peak_periods.loc[period, "power"],
                            )
                        ),
                        max=(self.bus_activation.loc[horizon.ph.dti, period]),
                    )
                },
                conversion_factors={self.bus_connected: 1},
            )
            for period in self.peak_periods.index
        }

        self.components.update(self.outflows)

        horizon.constraints.add_invest_costs(
            invest=(self.components[f"{self.name}_inflow_1"], self.components["bus"]),
            capex_spec=self.pois["s2g"].capex.spec,
            invest_type="flow",
        )
        horizon.constraints.add_invest_costs(
            invest=(
                self.components["bus"],
                self.components[f"{self.name}_outflow_{self.peak_periods.index[0]}"],
            ),
            capex_spec=self.pois["g2s"].capex.spec,
            invest_type="flow",
        )

        # The optimized sizes of the buses of all peakshaving intervals have to be the same as they technically
        # represent the same grid connection
        equal_investments = [{"in": self.components["bus"], "out": outflow} for outflow in self.outflows.values()]

        # If size of in- and outflow from and to the grid have to be the same size, add outflow investment(s)
        if self.expansion_equal:
            equal_investments.append(
                {
                    "in": self.components[f"{self.name}_inflow_1"],
                    "out": self.components["bus"],
                }
            )  # currently only works without peakshaving for inflows

        # add list of variables to the scenario constraints if list contains more than one element
        # lists with one element occur, if peakshaving is deactivated and grid sizes don't have to be equal
        if len(equal_investments) > 1:
            horizon.constraints.add_equal_invests(equal_investments)

    def get_horizon_results(self, horizon: simulation.PredictionHorizon):
        """
        post horizon method
        """
        self.sizes["g2s"].expansion = horizon.results[(self.components["bus"], list(self.outflows.values())[0])][
            "scalars"
        ]["invest"]
        self.sizes["s2g"].expansion = horizon.results[(list(self.inflows.values())[0], self.components["bus"])][
            "scalars"
        ]["invest"]

        self.flows.loc[horizon.ch.dti, "in"] = sum(
            [
                horizon.results[(inflow, self.components["bus"])]["sequences"]["flow"][horizon.ch.dti]
                for inflow in self.inflows.values()
            ]
        )
        self.flows.loc[horizon.ch.dti, "out"] = sum(
            [
                horizon.results[(self.components["bus"], outflow)]["sequences"]["flow"][horizon.ch.dti]
                for outflow in self.outflows.values()
            ]
        )

        # ToDo: apply peak powers
        # def get_peak_power(row):
        #     peak_power = max(
        #         row["power"],
        #         horizon.results[
        #             (
        #                 self.outflows[f"{self.name}_outflow_{row.name}"],
        #                 self.bus_connected,
        #             )
        #         ]["sequences"]["flow"][horizon.ch.dti].max(),
        #     )
        #     return peak_power
        #
        # self.peak_periods["power"] = self.peak_periods.apply(get_peak_power, axis=1)

    def calc_results_energies(self):
        super().calc_results_energies()
        # self.scenario.energies.loc[("sources", "pro"), :] += self.energies.loc["out", :]
        # self.scenario.energies.loc[("sinks", "del"), :] += self.energies.loc["in", :]
        # ToDo: fix this
        self.scenario.energies["sources"].add_energy(self.energies["out"])
        self.scenario.energies["sinks"].add_energy(self.energies["in"])


class GridMarket(ElectricBlock):
    def init_pois(self):
        super().init_pois()
        self.pois["g2s"] = eco.Evaluator.create(
            name="g2s",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex_spec=self.opex_spec_g2s,
            flow_name="out",
        )

        self.pois["s2g"] = eco.Evaluator.create(
            name="s2g",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex_spec=self.opex_spec_s2g,
            flow_name="in",
        )

    def __init__(self, name: str, scenario: simulation.PredictionHorizon, params, parent):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=params,
            parent=parent,
        )

    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None):
        """
        pre horizon method

        parent_bus
            |<---x----name_src
            |
            |----x--->name_snk
            |
        """

        self.components["src"] = solph.components.Source(
            outputs={
                self.parent.components["bus"]: solph.Flow(
                    nominal_capacity=self.pwr_g2s,
                    max=1 if self.pwr_g2s else None,
                    variable_costs=self.pois["g2s"].spec_ep_operation[horizon.ph.dti],
                )
            }
        )

        self.components["snk"] = solph.components.Sink(
            inputs={
                self.parent.components["bus"]: solph.Flow(
                    nominal_capacity=self.pwr_s2g,
                    max=1 if self.pwr_s2g else None,
                    variable_costs=(self.pois["s2g"].spec_ep_operation[horizon.ph.dti]),
                )
            }
        )

    def get_horizon_results(self, horizon: simulation.PredictionHorizon):
        """
        post horizon method
        """

        self.flows.loc[horizon.ch.dti, "in"] = horizon.results[(self.parent.components["bus"], self.components["snk"])][
            "sequences"
        ]["flow"][horizon.ch.dti]

        self.flows.loc[horizon.ch.dti, "out"] = horizon.results[
            (self.components["src"], self.parent.components["bus"])
        ]["sequences"]["flow"][horizon.ch.dti]


class StorageBlock(ElectricBlock):
    """
    abstract class
    """

    def init_pois(self):
        super().init_pois()
        self.pois["storage"] = eco.Evaluator.create(
            name="storage",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex_spec=self.capex_spec,
            consider_preexisting=self.capex_preexisting_storage,
            ls=self.ls,
            capex_ccr=self.ccr,
            mntex_spec=self.mntex_spec,
        )

        self.pois["in"] = eco.Evaluator.create(
            name="in",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex_spec=self.opex_spec,
            flow_name="in",
        )

        self.pois["out"] = eco.Evaluator.create(
            name="out",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            flow_name="out",
        )

        self.pois["bat_in"] = eco.Evaluator.create(
            name="bat_in",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            flow_name="bat_in",
        )

        self.pois["bat_out"] = eco.Evaluator.create(
            name="bat_out",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            flow_name="bat_out",
        )

        self.power_circles.append(("in", "out"))
        self.power_circles.append(("bat_in", "bat_out"))

    def init_states(self):
        super().init_states()
        for state in [
            "energy",
            "soc",
            "soh",
            "q_loss_cal",
            "q_loss_cyc",
            "soc_min",
            "soc_max",
        ]:
            self.states[state] = np.nan

    def __init__(
        self,
        name: str,
        scenario: simulation.Scenario,
        flow_apriori_names: list = None,
        params: dict = None,
        parent: BaseBlock | simulation.Scenario = None,
    ):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=flow_apriori_names,
            params=params,
            parent=parent,
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

        self.states.loc[:, "soc_min"] = (1 - self.states.loc[self.scenario.times.eval.start, "soh"]) / 2
        self.states.loc[:, "soc_max"] = 1 - ((1 - self.states.loc[self.scenario.times.eval.start, "soh"]) / 2)

        # initialization of aging model after all blocks are initialized to get temp from pv blocks
        self.aging_model = None

    def pre_scenario(self):
        super().pre_scenario()
        self.aging_model = bat.BatteryPackModel(self)

    def define_oemof_components(
        self,
        horizon: simulation.PredictionHorizon,
        params: dict = None,
    ):
        """
        pre horizon method
        x denotes the flow measurement point in results

        bus_connected   name_bus
             |             |
             |<-x-name_xc--|
             |             |<--->name_ess
             |-x-name_ess->|
             |             |

        """

        if params is None:
            raise ValueError(
                f'Block "{self.name}": Parameter "params" is required for StorageBlock method define_oemof_components()'
            )

        self.components["bus"] = solph.Bus()

        self.components["inflow"] = solph.components.Converter(
            inputs={
                self.bus_connected: solph.Flow(
                    nominal_capacity=params["inflow_nominal_capacity"],
                    max=params["inflow_max"],
                    fix=params["inflow_fix"],
                )
            },
            outputs={
                self.components["bus"]: solph.Flow(
                    variable_costs=self.scenario.cost_eps * -3  # incentivize charging of StorageBlocks vs. curtailment
                )
            },
            conversion_factors={self.components["bus"]: self.eff["chg_int"]},
        )

        self.components["outflow"] = solph.components.Converter(
            inputs={self.components["bus"]: solph.Flow()},
            outputs={
                self.bus_connected: solph.Flow(
                    nominal_capacity=params["outflow_nominal_capacity"],
                    max=params["outflow_max"],
                    fix=params["outflow_fix"],
                    variable_costs=self.scenario.cost_eps
                    * 4,  # disincentivize waste loop with inflow (sum must be positive)
                )
            },
            conversion_factors={self.bus_connected: self.eff["dis_int"]},
        )

        self.components["storage"] = solph.components.GenericStorage(
            inputs={
                self.components["bus"]: solph.Flow(variable_costs=self.pois["in"].spec_ep_operation[horizon.ph.dti])
            },
            outputs={self.components["bus"]: solph.Flow(variable_costs=self.scenario.cost_eps)},
            loss_rate=self.loss_rate_per_hour,
            balanced=params["storage_balanced"],
            initial_storage_level=self.states.loc[horizon.ph.start, ["soc", "soc_min", "soc_max"]].median(),
            # crate measured "outside" of conversion factor (efficiency)
            invest_relation_input_capacity=params["invest_relation_input_capacity"],
            invest_relation_output_capacity=params["invest_relation_output_capacity"],
            inflow_conversion_factor=np.sqrt(self.eff["storage_roundtrip"]),
            outflow_conversion_factor=np.sqrt(self.eff["storage_roundtrip"]),
            nominal_capacity=solph.Investment(
                ep_costs=self.pois["storage"].spec_ep_invest,
                existing=self.sizes["storage"].preexisting,
                maximum=self.sizes["storage"].expansion_max,
            ),
            max_storage_level=self.states.loc[horizon.ph.dti_extd, "soc_max"],
            min_storage_level=self.states.loc[horizon.ph.dti_extd, "soc_min"],
        )

        horizon.constraints.add_invest_costs(
            invest=(self.components["storage"],),
            capex_spec=self.pois["storage"].capex.spec,
            invest_type="storage",
        )

    def get_horizon_results(self, horizon: simulation.PredictionHorizon):
        """
        post horizon method
        """
        self.sizes["storage"].expansion = horizon.results[(self.components["storage"], None)]["scalars"]["invest"]

        self.flows.loc[horizon.ch.dti, "out"] = horizon.results[(self.components["outflow"], self.bus_connected)][
            "sequences"
        ]["flow"][horizon.ch.dti]
        self.flows.loc[horizon.ch.dti, "in"] = horizon.results[(self.bus_connected, self.components["inflow"])][
            "sequences"
        ]["flow"][horizon.ch.dti]

        self.flows.loc[horizon.ch.dti, "bat_out"] = horizon.results[
            (self.components["storage"], self.components["bus"])
        ]["sequences"]["flow"][horizon.ch.dti]
        self.flows.loc[horizon.ch.dti, "bat_in"] = horizon.results[
            (self.components["bus"], self.components["storage"])
        ]["sequences"]["flow"][horizon.ch.dti]

        self.states.loc[horizon.ch.dti_extd, "energy"] = horizon.results[(self.components["storage"], None)][
            "sequences"
        ]["storage_content"][horizon.ch.dti_extd]
        # divide by 0 (size=0) -> pandas returns NaN -> SOC init = NaN in next horizon -> pyomo fails -> fillna(0)
        self.states.loc[horizon.ch.dti_extd, "soc"] = (
            self.states.loc[horizon.ch.dti_extd, "energy"] / self.sizes["storage"].total
        ).fillna(0)

        self.aging_model.age(horizon=horizon)

    def calc_results_flows(self):
        super().calc_results_flows()
        # ToDo: what is bat_total required for?
        self.flows["bat_total"] = self.flows.get(key="bat_out", default=0) - self.flows.get(key="bat_in", default=0)


class StationaryBattery(StorageBlock):
    def __init__(
        self,
        name: str,
        scenario: simulation.Scenario,
    ):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=None,
            parent=scenario,
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

    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None):
        self.bus_connected = self.scenario.block_registry.get("TopLevelBlock", {})["core"].components[self.system]
        params = {
            "inflow_nominal_capacity": None,
            "outflow_nominal_capacity": None,
            "inflow_max": None,
            "outflow_max": None,
            "inflow_fix": None,
            "outflow_fix": None,
            "invest_relation_input_capacity": self.crate_chg,
            "invest_relation_output_capacity": self.crate_dis,
            "storage_balanced": True if self.scenario.strategy == "go" else False,
        }
        super().define_oemof_components(horizon, params)


class Fleet(SinkBlock):
    def init_pois(self):
        super().init_pois()
        self.pois["f2s"] = eco.Evaluator.create(
            name="f2s",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex_spec=self.opex_spec_f2s,
            flow_name="out",
        )

        self.pois["s2f"] = eco.Evaluator.create(
            name="s2f",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex_spec=self.opex_spec_s2f,
            flow_name="in",
        )

    def __init__(self, name: str, scenario: simulation.Scenario):
        super().__init__(
            name=name,
            scenario=scenario,
            flow_apriori_names=None,
            params=None,
            parent=scenario,
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
                self.scenario.paths.create_result_path(suffix=f"{self.scenario.name}_{self.name}_demand.csv")
                if not self.scenario.settings.largescalemode
                else None
            )
            self.demand.from_usecases(
                path_usecases=self.scenario.paths.input
                / utils.set_extension(filename=self.filename, default_extension=".csv"),
                path_timeframe_mapper=self.scenario.paths.input / f"{self.filename_mapper}.py",
                path_demand=path_demand,
                key_timeframe_mapper=self.name,
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
            df = utils.read_timeseries_csv(
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

            cols = df.columns  # save orignal column sorting to apply after resampling
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

    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None):
        """
        pre horizon method

        x denotes the flow measurement point in results
        xc denotes ac or dc, depending on the parameter 'system'

        bus_connected        name_bus
          |<----name_outflow--x-|---(ElectricFleetUnit Instance)
          |                     |
          |-x----name_inflow--->|---(ElectricFleetUnit Instance)
          |                     |
          |                     |   (CombustionVehicle Instance)
        """

        self.components["bus"] = solph.Bus()
        self.bus_connected = self.scenario.block_registry.get("TopLevelBlock", {})["core"].components[self.system]

        self.components["inflow"] = solph.components.Converter(
            inputs={
                self.bus_connected: solph.Flow(
                    variable_costs=self.pois["s2f"].spec_ep_operation[horizon.ph.dti],
                    nominal_capacity=self.pwr_lim_s2f,
                    # default value for max is 1; not explicitly set to ensure compatibility with nominal_capacity=None
                )
            },
            outputs={self.components["bus"]: solph.Flow()},
            conversion_factors={self.components["bus"]: 1},
        )

        self.components["outflow"] = solph.components.Converter(
            inputs={
                self.components["bus"]: solph.Flow(
                    variable_costs=self.pois["f2s"].spec_ep_operation[horizon.ph.dti],
                    nominal_capacity=self.pwr_lim_f2s,
                    # default value for max is 1; not explicitly set to ensure compatibility with nominal_capacity=None
                )
            },
            outputs={self.bus_connected: solph.Flow(variable_costs=self.scenario.cost_eps)},
            conversion_factors={self.bus_connected: 1},
        )

    def get_horizon_results(self, horizon: simulation.PredictionHorizon):
        """
        post horizon method
        """
        self.flows.loc[horizon.ch.dti, "out"] = horizon.results[(self.components["outflow"], self.bus_connected)][
            "sequences"
        ]["flow"][horizon.ch.dti]
        self.flows.loc[horizon.ch.dti, "in"] = horizon.results[(self.bus_connected, self.components["inflow"])][
            "sequences"
        ]["flow"][horizon.ch.dti]


class SubFleet(NonElectricBlock):
    def __init__(self, name: str, scenario: simulation.Scenario, parent):
        # subfleet parameters contain FleetUnit parameters -> split parameters for FleetUnits and SubFleet
        params = scenario.parameters.loc[name]
        params_subfleet = {key: params.pop(key) if key in params else None for key in ["num", "type_unit", "rex"]}

        super().__init__(name=name, scenario=scenario, params=params_subfleet, parent=parent)

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

        if getattr(self, "invest", False) and self.data_source in [
            "usecases",
            "demand",
        ]:
            self.scenario.logger.Error(
                f'Subfleet "{self.name}": investment not implemented for data source "{self.data_source}"'
            )

    def pre_scenario(self):
        self.log = self.parent.log.loc[:, self.parent.log.columns.get_level_values(0).str.contains(self.name)]
        super().pre_scenario()


class FleetUnit:
    def init_pois(self):
        self.pois["glider"] = eco.Evaluator.create(
            name="glider",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex_fix=self.capex_fix_glider,
            ls=self.ls,
            capex_ccr=self.ccr,
            consider_preexisting=self.capex_preexisting_glider,
            mntex_fix=self.mntex_fix_glider,
            # ToDo: add dist and time based opex and crev configs
            # opex_config_fleetunit=dict(dist=self.opex_spec_dist),
            # crev_config_fleetunit=dict(dist=self.crev_spec_dist, time=self.crev_spec_time),
        )

    def __init__(self):
        self.log = None

    def pre_scenario(self):
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


class ElectricFleetUnit(StorageBlock, FleetUnit):
    """
    abstract class
    """

    def init_pois(self):
        StorageBlock.init_pois(self)
        FleetUnit.init_pois(self)

        self.pois["charger"] = eco.Evaluator.create(
            name="charger",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            capex_fix=self.capex_fix_charger,
            ls=self.ls,
            capex_ccr=self.ccr,
            consider_preexisting=self.capex_preexisting_charger,
        )

        self.pois["ext_ac"] = eco.Evaluator.create(
            name="ext_ac",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex_spec=self.opex_spec_ext_ac,
            flow_name="ext_ac",
        )

        self.pois["ext_dc"] = eco.Evaluator.create(
            name="ext_dc",
            eco=self.scenario.eco_params,
            data_dir=self.scenario.paths.input,
            opex_spec=self.opex_spec_ext_dc,
            flow_name="ext_dc",
        )

    def __init__(self, name: str, scenario: simulation.Scenario, parent: SubFleet, params: dict):
        StorageBlock.__init__(
            self=self,
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
        )

        FleetUnit.__init__(self=self)

        self.apriori = True if self.mode_scheduling in self.scenario.apriori_lvls else False

        if any([size.invest for size in self.sizes.values()]) and self.mode_scheduling in self.scenario.apriori_lvls:
            raise ValueError(
                f'ElectricFleetUnit "{self.name}": size optimization not '
                f"implemented for a priori integration levels: {self.scenario.apriori_lvls}"
            )

    def initialize_efficiencies(self):
        self.eff["chg_int"] = {"ac": self.eff_chg_ac, "dc": self.eff_chg_dc}[self.parent.parent.system]
        self.eff["dis_int"] = {"ac": self.eff_dis_ac, "dc": self.eff_dis_dc}[self.parent.parent.system]
        super().initialize_efficiencies()

    def pre_scenario(self):
        StorageBlock.pre_scenario(self=self)
        FleetUnit.pre_scenario(self=self)

    def define_oemof_components(self, horizon: simulation.PredictionHorizon, params: dict = None):
        """
        pre horizon method

        parent.parent_bus     name_bus
            |<--x--name_fleet---|<-x->name_storage (handled in StorageBlock)
            |                   |
            |---x--fleet_name-->|-->name_snk (handled in StorageBlock)
            |                   |
            |                   |<--name_ext_ac-x- (external charging AC)
            |                   |
            |                   |<--name_ext_dc-x- (external charging DC)
            |
        """

        # region calc minimum soc targets before usage and max soc for myopic optimization
        dsoc_ph = self.log.loc[horizon.ph.dti, "dsoc"]
        if (self.scenario.strategy == "rh") and (self.mode_scheduling == "oc") and isinstance(self, ElectricVehicle):
            soc_min_hor = dsoc_ph.mask(cond=dsoc_ph > 0, other=dsoc_ph + self.dsoc_buffer).clip(
                lower=self.states.loc[horizon.ph.dti_extd, "soc_min"],
                upper=self.states.loc[horizon.ph.dti_extd, "soc_max"],
            )
        elif (self.scenario.strategy == "rh") and (self.mode_scheduling == "oc") and isinstance(self, MobileBattery):
            soc_min_hor = dsoc_ph.mask(cond=dsoc_ph > 0, other=self.soc_target).clip(
                lower=self.states.loc[horizon.ph.dti_extd, "soc_min"],
                upper=self.states.loc[horizon.ph.dti_extd, "soc_max"],
            )
        else:  # a priori or global optimization
            soc_min_hor = self.states.loc[horizon.ph.dti_extd, "soc_min"]
        self.states.update({"soc_min": soc_min_hor.astype("float64")})
        # endregion

        self.bus_connected = self.parent.parent.components["bus"]

        params = {
            "inflow_nominal_capacity": self.pwr_chg_max,
            "outflow_nominal_capacity": self.pwr_dis_max * self.eff["dis_int"],
            "inflow_max": None if self.apriori else self.log.loc[horizon.ph.dti, "atbase"].astype(int),
            "outflow_max": None if self.apriori else self.log.loc[horizon.ph.dti, "atbase"].astype(int),
            "inflow_fix": self.flows_apriori.loc[horizon.ph.dti, "p_int_chg"] if self.apriori else None,
            "outflow_fix": self.flows_apriori.loc[horizon.ph.dti, "p_int_dis"] if self.apriori else None,
            "invest_relation_input_capacity": None,
            "invest_relation_output_capacity": None,
            "storage_balanced": False,
        }

        super().define_oemof_components(horizon=horizon, params=params)

        self.components["snk"] = solph.components.Sink(
            inputs={
                self.components["bus"]: solph.Flow(nominal_capacity=1, fix=self.log.loc[horizon.ph.dti, "consumption"])
            }
        )

        self.components["bus_ext_ac"] = solph.Bus()

        self.components["src_ext_ac"] = solph.components.Source(
            outputs={
                self.components["bus_ext_ac"]: solph.Flow(
                    nominal_capacity=self.pwr_ext_ac_max,
                    max=None if self.apriori else self.log.loc[horizon.ph.dti, "atac"].astype(int),
                    fix=self.flows_apriori.loc[horizon.ph.dti, "p_ext_ac_chg"] if self.apriori else None,
                    variable_costs=self.pois["ext_ac"].spec_ep_operation[horizon.ph.dti],
                )
            }
        )

        self.components["conv_ext_ac"] = solph.components.Converter(
            inputs={self.components["bus_ext_ac"]: solph.Flow()},
            outputs={self.components["bus"]: solph.Flow()},
            conversion_factors={self.components["bus"]: self.eff["chg_ac"]},
        )

        self.components["bus_ext_dc"] = solph.Bus()

        self.components["src_ext_dc"] = solph.components.Source(
            outputs={
                self.components["bus_ext_dc"]: solph.Flow(
                    nominal_capacity=self.pwr_ext_dc_max,
                    max=None if self.apriori else self.log.loc[horizon.ph.dti, "atdc"].astype(int),
                    fix=self.flows_apriori.loc[horizon.ph.dti, "p_ext_dc_chg"] if self.apriori else None,
                    variable_costs=self.pois["ext_dc"].spec_ep_operation[horizon.ph.dti],
                )
            }
        )

        self.components["conv_ext_dc"] = solph.components.Converter(
            inputs={self.components["bus_ext_dc"]: solph.Flow()},
            outputs={self.components["bus"]: solph.Flow()},
            conversion_factors={self.components["bus"]: 1},  # billed energy is already dc in external dc charging
        )

    def get_horizon_results(self, horizon: simulation.PredictionHorizon):
        """
        post horizon method
        """

        self.flows.loc[horizon.ch.dti, "ext_ac"] = horizon.results[
            (self.components["bus_ext_ac"], self.components["conv_ext_ac"])
        ]["sequences"]["flow"][horizon.ch.dti]
        self.flows.loc[horizon.ch.dti, "ext_dc"] = horizon.results[
            (self.components["bus_ext_dc"], self.components["conv_ext_dc"])
        ]["sequences"]["flow"][horizon.ch.dti]

        super().get_horizon_results(horizon=horizon)


class CombustionVehicle(NonElectricBlock, FleetUnit):
    def init_pois(self):
        NonElectricBlock.init_pois(self=self)
        FleetUnit.init_pois(self=self)

    def __init__(self, name: str, scenario: simulation.Scenario, parent: SubFleet, params: dict):
        NonElectricBlock.__init__(self=self, name=name, scenario=scenario, params=params, parent=parent)

        FleetUnit.__init__(self=self)

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

    def pre_scenario(self):
        NonElectricBlock.pre_scenario(self=self)
        FleetUnit.pre_scenario(self=self)


class ElectricVehicle(ElectricFleetUnit):
    """
    dummy class to enable tracking
    """

    pass


class MobileBattery(ElectricFleetUnit):
    def __init__(self, name: str, scenario: simulation.Scenario, parent: SubFleet, params: dict):
        self.opex_spec_dist = 0.0  # no distance based opex for mobile battery
        self.opex_spec_time = 0.0  # no distance based opex for mobile battery
        super().__init__(name=name, scenario=scenario, parent=parent, params=params)
