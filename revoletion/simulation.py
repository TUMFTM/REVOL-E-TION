#!/usr/bin/env python3

import logging
import types
from typing import override

from . import blocks, optimization
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


class OptimizationHorizonResultProcessor(blocks.BlockVisitor[None]):
    def __init__(self, optimization_result: optimization.OptimizationResult) -> None:
        self._optimization_result = optimization_result

    @classmethod
    def collect_optimization_results(
        cls,
        optimization_result: optimization.OptimizationResult,
        scenario: scn.Scenario,
        horizon: scn.TimeSettings,
        horizon_index: int,
    ) -> None:
        visitor = cls(optimization_result)
        for block in scenario.block_registry.get("TopLevelBlock", {}).values():
            visitor.visit_block(block, horizon=horizon, horizon_index=horizon_index)

    @override
    def visit_block(self, block: blocks.BaseBlock, horizon: scn.TimeSettings, horizon_index: int) -> None:
        # Always traverse to children even for NonElectricBlock. This is necessary, since
        # SubFleet is a NonElectricBlock, but it might have electric subblocks.
        for subblock in block.subblocks.values():
            self.visit_block(subblock, horizon=horizon, horizon_index=horizon_index)

        # For `NonElectricBlock` no further processing should be done, since they have no power flows
        # and also no investment option.
        if not isinstance(block, blocks.ElectricBlock):
            return

        investments = self._optimization_result.get_capex(block)
        for size_name, investment in investments.items():
            block.sizes[size_name].expansion = investment

        # `GridConnection` needs some special power flow extraction.
        if isinstance(block, blocks.GridConnection):
            return self.visit_grid_connection(block, horizon)

        power_flows = self._optimization_result.get_power_flow(block, horizon.dti)

        for power_flow_name, power_flow in power_flows.items():
            block.flows.loc[horizon.dti, power_flow_name] = power_flow

        if isinstance(block, blocks.StorageBlock):
            self.visit_storage_block(block, horizon, horizon_index)

    def visit_grid_connection(self, block: blocks.GridConnection, horizon: scn.TimeSettings) -> None:
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

    def visit_storage_block(self, block: blocks.StorageBlock, horizon: scn.TimeSettings, horizon_index: int) -> None:
        stored_energy = self._optimization_result.get_stored_energy(block, horizon.dti_extd)
        block.states.loc[horizon.dti_extd, "energy"] = stored_energy
        block.states.loc[horizon.dti_extd, "soc"] = (stored_energy / block.sizes["storage"].total).fillna(0)

        if not block.aging:
            block.states.loc[horizon.end, "soh"] = block.states.loc[horizon.start, "soh"]
            return

        # This is a small hack to make the battery aging model compatible with the new results extraction API.
        # TODO: decouple the battery aging model from the horizon and remove this hack.
        comp_horizon = types.SimpleNamespace(ch=horizon, index=horizon_index)
        block.aging_model.age(comp_horizon)


class OptimizationHorizon:
    index: int
    scenario: scn.Scenario

    _logger: logging.Logger

    def __init__(self, index: int, scenario: scn.Scenario, logger: logging.Logger):
        self.index = index
        self.scenario = scenario

        self._logger = logger

        # region time and data generation and slicing
        start = self.scenario.times.sim.start + (self.index * self.scenario.len_ch)
        self.ph = scn.TimeSettings.create_from_start_timestamp(
            start=start,
            timestep=self.scenario.timestep.td,
            end=min(start + self.scenario.len_ph, self.scenario.times.sim.end),
        )

        self.ch = scn.TimeSettings.create_from_start_timestamp(
            start=start,
            timestep=self.scenario.timestep.td,
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
        solver = optimization.Solver(self.scenario.settings.solver)

        optimization_model_config = optimization.OptimizationModelConfig(
            cost_eps=self.scenario.cost_eps,
            debug=self.scenario.settings.debugmode,
            solver=solver,
            invest=True,
        )

        optimization_model = optimization.OemofOptimizationModel.from_revoletion_scenario(
            scenario=self.scenario,
            horizon=self.ph,
            logger=self._logger,
            config=optimization_model_config,
        )

        status, optimization_result = optimization_model.optimize()
        if status != optimization.OptimizationStatus.OPTIMAL:
            raise OptimizationError(
                f"Scenario failed: {status.value}. Enable debug mode for more information.",
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

        OptimizationHorizonResultProcessor.collect_optimization_results(
            optimization_result, self.scenario, horizon=self.ch, horizon_index=self.index
        )
