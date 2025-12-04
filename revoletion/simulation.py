#!/usr/bin/env python3

import logging
import types
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from typing_extensions import override

from . import blocks, optimization, rl, utils
from . import scenario as scn

_LOGGER = logging.getLogger(__name__)


class OptimizationError(Exception):
    def __init__(
        self, msg: str, optimization_horizon_idx: int | None = None, optimization_horizon_num: int | None = None
    ) -> None:
        """
        Create a new OptimizationError.

        :param msg: The error message.
        :param optimization_horizon_idx: Optionally provide the index of the optimization horizon that failed, to include more information in the error message.
        :param optimization_horizon_num: Optionally provide the total number of the optimization horizon, to include more information in the error message.

        :returns: The new OptimizationError.
        """
        self.optimization_horizon_idx = optimization_horizon_idx
        self.optimization_horizon_num = optimization_horizon_num

        if self.optimization_horizon_idx is not None and self.optimization_horizon_num is not None:
            msg = f"Horizon {self.optimization_horizon_idx} of {self.optimization_horizon_num} - {msg}"

        super().__init__(msg)


class _ExpansionResultProcessor(blocks.BlockVisitor[None]):
    def __init__(self, optimization_result: optimization.OptimizationResult) -> None:
        self._optimization_result = optimization_result

    @classmethod
    def collect_expansions(
        cls,
        optimization_result: optimization.OptimizationResult,
        scenario: scn.Scenario,
        horizon: utils.TimeSettings,
    ) -> None:
        """
        Collect the optimization results and write them back to each individual block.

        :param optimization_result: The optimization results from an `OptimizationHorizon`.
        :param scenario: The scenario which was optimized.
        :param horizon: The time horizon of the optimization.
        """
        visitor = cls(optimization_result)
        for block in scenario.block_registry.get("TopLevelBlock", {}).values():
            visitor.visit_block(block, horizon=horizon)

    @override
    def visit_block(self, block: blocks.BaseBlock, horizon: utils.TimeSettings) -> None:
        # Always traverse to children even for NonElectricBlock. This is necessary, since
        # SubFleet is a NonElectricBlock, but it might have electric subblocks.
        for subblock in block.subblocks.values():
            self.visit_block(subblock, horizon=horizon)

        # For `NonElectricBlock` no further processing should be done, since they have no power flows
        # and also no investment option.
        if not isinstance(block, blocks.ElectricBlock):
            return

        # Investments are directly written back to each block.
        expansions = self._optimization_result.get_expansion(block)
        for size_name, expansion in expansions.items():
            block.sizes[size_name].expansion = expansion


class _BatteryHealthResultProcessor(blocks.BlockVisitor[None]):
    def __init__(self, optimization_result: optimization.OptimizationResult) -> None:
        self._optimization_result = optimization_result

    @classmethod
    def collect_health_status(
        cls,
        optimization_result: optimization.OptimizationResult,
        scenario: scn.Scenario,
        horizon: utils.TimeSettings,
        horizon_index: int,
    ) -> None:
        """
        Collect the optimization results and write them back to each individual block.

        :param optimization_result: The optimization results from an `OptimizationHorizon`.
        :param scenario: The scenario which was optimized.
        :param horizon: The time horizon of the optimization.
        :param horizon_index: The index of the optimization horizon.
        """
        visitor = cls(optimization_result)
        for block in scenario.block_registry.get("TopLevelBlock", {}).values():
            # `horizon_index` must be passed down, because the battery aging model currently uses it.
            # TODO: Ideally, the aging model would be independent of this and the `horizon_index` can be removed.
            visitor.visit_block(block, horizon=horizon, horizon_index=horizon_index)

    @override
    def visit_block(self, block: blocks.BaseBlock, horizon: utils.TimeSettings, horizon_index: int) -> None:
        # Always traverse to children even for NonElectricBlock. This is necessary, since
        # SubFleet is a NonElectricBlock, but it might have electric subblocks.
        for subblock in block.subblocks.values():
            self.visit_block(subblock, horizon=horizon, horizon_index=horizon_index)

        # For `NonElectricBlock` no further processing should be done, since they have no power flows
        # and also no investment option.
        if not isinstance(block, blocks.StorageBlock):
            return

        self.visit_storage_block(block, horizon, horizon_index)

    def visit_storage_block(self, block: blocks.StorageBlock, horizon: utils.TimeSettings, horizon_index: int) -> None:
        if not block.aging:
            block.states.loc[horizon.end, "soh"] = block.states.loc[horizon.start, "soh"]
            return

        # This is a small hack to make the battery aging model compatible with the new results extraction API.
        # TODO: decouple the battery aging model from the horizon and remove this hack.
        comp_horizon = types.SimpleNamespace(ch=horizon, index=horizon_index)
        block.aging_model.age(comp_horizon)


class _PowerFlowResultProcessor(blocks.BlockVisitor[None]):
    """
    Process the results of an optimization horizon execution.

    This transforms the results produced by an `OptimizationModel` into flows and investments.
    The results are written to each block, to expose the data to subsequent optimizations or the final result collection.
    """

    def __init__(self, optimization_result: optimization.OptimizationResult) -> None:
        self._optimization_result = optimization_result

    @classmethod
    def collect_power_flows(
        cls,
        optimization_result: optimization.OptimizationResult,
        scenario: scn.Scenario,
        horizon: utils.TimeSettings,
    ) -> None:
        """
        Collect the optimization results and write them back to each individual block.

        :param optimization_result: The optimization results from an `OptimizationHorizon`.
        :param scenario: The scenario which was optimized.
        :param horizon: The time horizon of the optimization.
        :param horizon_index: The index of the optimization horizon.
        """
        visitor = cls(optimization_result)
        for block in scenario.block_registry.get("TopLevelBlock", {}).values():
            # `horizon_index` must be passed down, because the battery aging model currently uses it.
            # TODO: Ideally, the aging model would be independent of this and the `horizon_index` can be removed.
            visitor.visit_block(block, horizon=horizon)

    @override
    def visit_block(self, block: blocks.BaseBlock, horizon: utils.TimeSettings) -> None:
        # Always traverse to children even for NonElectricBlock. This is necessary, since
        # SubFleet is a NonElectricBlock, but it might have electric subblocks.
        for subblock in block.subblocks.values():
            self.visit_block(subblock, horizon=horizon)

        # For `NonElectricBlock` no further processing should be done, since they have no power flows.
        if not isinstance(block, blocks.ElectricBlock):
            return

        # `GridConnection` needs some special power flow extraction to handle peak-periods.
        if isinstance(block, blocks.GridConnection):
            return self.visit_grid_connection(block, horizon)

        power_flows = self._optimization_result.get_power_flow(block, horizon.dti)

        for power_flow_name, power_flow in power_flows.items():
            block.flows.loc[horizon.dti, power_flow_name] = power_flow

        if isinstance(block, blocks.StorageBlock):
            self.visit_storage_block(block, horizon)

    def visit_grid_connection(self, block: blocks.GridConnection, horizon: utils.TimeSettings) -> None:
        """
        Collect the results for a `GridConnection`.

        Aggregates the individual results for each peak-period.
        """
        power_flows = self._optimization_result.get_power_flow(block, horizon.dti)

        outflows = {flow_name: flow for flow_name, flow in power_flows.items() if flow_name.startswith("out")}
        outflow = sum(outflows.values())
        inflow = sum(flow for flow_name, flow in power_flows.items() if flow_name.startswith("in"))

        block.flows.loc[horizon.dti, "out"] = outflow
        block.flows.loc[horizon.dti, "in"] = inflow

        def get_peak_power(row):
            peak_power = max(row["power"], outflows[f"outflow_{row.name}"].max())
            return peak_power

        block.peak_periods["power"] = block.peak_periods.apply(get_peak_power, axis=1)

    def visit_storage_block(self, block: blocks.StorageBlock, horizon: utils.TimeSettings) -> None:
        # TODO: dti or dti_extd??
        stored_energy = self._optimization_result.get_stored_energy(block, horizon.dti)
        block.states.loc[horizon.dti_extd, "energy"] = stored_energy
        block.states.loc[horizon.dti_extd, "soc"] = (stored_energy / block.sizes["storage"].total).fillna(0)


@dataclass
class SimulationSettings:
    solver: optimization.Solver = optimization.Solver.GUROBI
    backend: optimization.OptimizationBackend = optimization.OptimizationBackend.OEMOF
    n_processes: int = 1
    largescalemode: bool = False
    debugmode: bool = False
    rerun_infeasible: bool = True
    key_solcast_api: str | None = None


class OptimizationHorizon:
    index: int
    scenario: scn.Scenario

    _logger: logging.Logger

    def __init__(self, index: int, scenario: scn.Scenario, settings: SimulationSettings, logger: logging.Logger):
        self.index = index
        self.scenario = scenario
        self._settings = settings

        self._logger = logger

        # region time and data generation and slicing
        start = self.scenario.times.sim.start + (self.index * self.scenario.len_ch)
        self.ph = utils.TimeSettings.create_from_start_timestamp(
            start=start,
            timestep=self.scenario.timestep,
            end=min(start + self.scenario.len_ph, self.scenario.times.sim.end),
        )

        self.ch = utils.TimeSettings.create_from_start_timestamp(
            start=start,
            timestep=self.scenario.timestep,
            end=min(start + self.scenario.len_ch, self.scenario.times.eval.end),
        )

        # Display logger message if PH exceeds simulation end time and has to be truncated
        if self.ph.duration < self.scenario.len_ph:
            self._logger.info(msg="Prediction Horizon truncated to simulation end time")

        self._logger.info("Start: %s - CH end: %s - PH end: %s", self.ph.start, self.ch.end, self.ph.end)

        # if apriori power scheduling is necessary, calculate power schedules:
        if self.scenario.scheduler:
            self._logger.debug("Calculating power schedules for commodities with rulebased charging strategies")
            self.scenario.scheduler.calc_ph_schedule(self.ph)
        # endregion

    def execute(self) -> None:
        """
        Perform the concrete optimization across an optimization horizon.
        """

        optimization_problem_config = optimization.OptimizationProblemConfig(
            cost_eps=self.scenario.cost_eps,
            debug=self._settings.debugmode,
            solver=self._settings.solver,
            invest=True,
        )

        optimization_problem = optimization.create_optimization_problem(
            backend=self._settings.backend,
            scenario=self.scenario,
            horizon=self.ph,
            logger=self._logger,
            config=optimization_problem_config,
        )

        status, optimization_result = optimization_problem.solve()

        if status != optimization.OptimizationStatus.OPTIMAL:
            raise OptimizationError(
                f"Scenario failed: {status}. Enable debug mode for more information.",
                optimization_horizon_idx=self.index,
                optimization_horizon_num=self.scenario.nhorizons,
            )

        if optimization_result is None:
            raise OptimizationError(
                "Optimization failed: No optimization result, even though the solver signaled an optimal result. This is a bug.",
                optimization_horizon_idx=self.index,
                optimization_horizon_num=self.scenario.nhorizons,
            )

        if self.scenario.nhorizons == 1:  # Don't store objective for multiple horizons in scenario (most RH scenarios)
            self.scenario.objective_opt = optimization_result.get_objective()

        _BatteryHealthResultProcessor.collect_health_status(
            optimization_result, self.scenario, horizon=self.ch, horizon_index=self.index
        )
        _PowerFlowResultProcessor.collect_power_flows(
            optimization_result,
            self.scenario,
            horizon=self.ch,
        )
        _ExpansionResultProcessor.collect_expansions(
            optimization_result,
            self.scenario,
            horizon=self.ch,
        )


@dataclass
class ControlSettings:
    n_processes: int = 1
    agent_algorithm: rl.AgentAlgorithm = rl.AgentAlgorithm.PPO
    train_timesteps: int | None = None
    models_path: Path | None = None
    debugmode: bool = False


_DEFAULT_TRAIN_TIMESTEPS = 10_000


class ControlScenarioFactory:
    def __init__(self, paths: scn.SimulationPaths) -> None:
        self._paths = paths
        self.scenario_name = self._paths.scenario.stem

        scenario_parameters = utils.read_scenario_from_file(self._paths.scenario)

        self._scenario_parameters = scenario_parameters[self.scenario_name]

        self._location = utils.Location.create_from_lat_lon(
            latitude=self._scenario_parameters.loc["scenario", "latitude"],
            longitude=self._scenario_parameters.loc["scenario", "longitude"],
            logger=_LOGGER,
        )

    def create_scenario(self) -> scn.Scenario:
        return scn.Scenario(
            self._paths,
            scn.ScenarioSettings(),
            name=self.scenario_name,
            parameters=self._scenario_parameters,
            location=self._location,
        )


class ControlHorizon:
    def __init__(
        self,
        scenario_factory: ControlScenarioFactory,
        settings: ControlSettings,
        logger: logging.Logger | None = None,
    ) -> None:
        self._scenario_factory = scenario_factory
        self._settings = settings
        self._logger = logger or _LOGGER

    def execute(self, plot=True) -> None:
        scenario = self._scenario_factory.create_scenario()

        agent = None
        if self._settings.models_path is not None:
            self._settings.models_path.mkdir(exist_ok=True)
            agent = rl.load_agent(
                self._settings.agent_algorithm, self._scenario_factory.scenario_name, self._settings.models_path
            )

        if agent is None:
            agent_config = rl.AgentConfig.default_for_algorithm(self._settings.agent_algorithm)
            agent_config.tensorboard_log = "/tmp/revol"

            train_horizon = utils.TimeSettings.create_from_start_timestamp(
                start=scenario.times.sim.start,
                timestep=scenario.timestep,
                end=scenario.times.sim.start + scenario.len_ph,
            )

            self._logger.info(
                f"Training agent '{self._settings.agent_algorithm}' on scenario '{self._scenario_factory.scenario_name}'"
            )
            agent = rl.train(
                self._settings.agent_algorithm,
                self._scenario_factory.create_scenario,
                horizon=train_horizon,
                n_proc=self._settings.n_processes,
                config=agent_config,
                total_timesteps=self._settings.train_timesteps or _DEFAULT_TRAIN_TIMESTEPS,
            )

            if self._settings.models_path is not None:
                rl.save_agent(agent, self._scenario_factory.scenario_name, self._settings.models_path)

        eval_horizon = utils.TimeSettings.create_from_start_timestamp(
            start=scenario.times.sim.start + scenario.len_ph,
            timestep=scenario.timestep,
            end=scenario.times.sim.start + scenario.len_ph + scenario.len_ch,
        )
        self._logger.info(f"Evaluating agent '{self._settings.agent_algorithm}'")
        reward, optimization_result = rl.evaluate_with_agent(scenario, agent, eval_horizon)
        self._logger.info(f"Agent got a reward of {reward}")
        if optimization_result is None:
            self._logger.error("Evaluation failed")
            return

        _PowerFlowResultProcessor.collect_power_flows(
            optimization_result,
            scenario,
            eval_horizon,
        )

        for block in scenario.block_registry.get("TopLevelBlock", {}).values():
            block.post_scenario()

        result_timeseries = blocks.TimeseriesCollectionBlockVisitor().collect_timeseries(
            scenario.block_registry, eval_horizon
        )
        result_timeseries_aggregated = pd.concat(result_timeseries, axis=1)
        result_timeseries_aggregated.to_csv(scenario.paths.create_result_path(suffix=f"{scenario.name}_results_ts.csv"))
        if plot:
            scenario.generate_and_save_plot()
