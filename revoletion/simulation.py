#!/usr/bin/env python3

import logging
import types
from dataclasses import dataclass
from typing import override

from revoletion import scenario as scn

from . import blocks, optimization, time
from . import logger as logger_fcs

_LOGGER = logging.getLogger(__name__)


class OptimizationError(Exception):
    def __init__(
        self, msg: str, prediction_horizon_idx: int | None = None, prediction_horizon_num: int | None = None
    ) -> None:
        """
        Create a new OptimizationError.

        :param msg: The error message.
        :param prediction_horizon_idx: Optionally provide the index of the prediction horizon that failed, to include more information in the error message.
        :param prediction_horizon_num: Optionally provide the total number of the prediction horizon, to include more information in the error message.

        :returns: The new OptimizationError.
        """
        self.prediction_horizon_idx = prediction_horizon_idx
        self.prediction_horizon_num = prediction_horizon_num

        if self.prediction_horizon_idx is not None and self.prediction_horizon_num is not None:
            msg = f"Horizon {self.prediction_horizon_idx} of {self.prediction_horizon_num} - {msg}"

        super().__init__(msg)


class _OptimizationHorizonResultProcessor(blocks.BlockVisitor[None]):
    def __init__(self, optimization_result: optimization.OptimizationResult) -> None:
        self._optimization_result = optimization_result

    @classmethod
    def collect_optimization_results(
        cls,
        optimization_result: optimization.OptimizationResult,
        scenario: scn.Scenario,
        horizon: time.TimeFrame,
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
    def visit_block(self, block: blocks.BaseBlock, horizon: time.TimeFrame, horizon_index: int) -> None:
        # Always traverse to children even for NonElectricBlock. This is necessary, since
        # SubFleet is a NonElectricBlock, but it might have electric subblocks.
        for subblock in block.subblocks.values():
            self.visit_block(subblock, horizon=horizon, horizon_index=horizon_index)

        # For `NonElectricBlock` no further processing should be done, since they have no power flows
        # and also no investment option.
        if not isinstance(block, blocks.ElectricBlock):
            return

        # Investments are directly written back to each block.
        expansions = self._optimization_result.get_expansion(block)
        for size_name, expansion in expansions.items():
            block.sizes[size_name].expansion = expansion

        power_flows = self._optimization_result.get_power_flow(block, horizon.dti)

        for power_flow_name, power_flow in power_flows.items():
            block.flows.loc[horizon.dti, power_flow_name] = power_flow

        if isinstance(block, blocks.StorageBlock):
            self.visit_storage_block(block, horizon, horizon_index)
        elif isinstance(block, blocks.GridConnection):
            self.visit_grid_connection(block, horizon)

    def visit_grid_connection(self, block: blocks.GridConnection, horizon: time.TimeFrame) -> None:
        """
        Collect the results for a `GridConnection`.

        Aggregates the individual results for each peak-period.
        """
        # get peak powers per peak interval
        block.peak_periods.update(
            {
                "peak_power": {
                    period.label: max(
                        # this leads to inconsistencies for dti_sim != dti_eval if peak power occurs after dti_eval
                        block.flows.loc[horizon.dti, "out"][block.peak_periods_activation[period.label]]
                        .resample(block.peak_period_measurement.freqstr)
                        .mean()
                        .max(),
                        period.peak_power,
                    )
                    for period in block.peak_periods.itertuples()
                    if period.label in block.peak_storages.keys()
                }
            }
        )

    def visit_storage_block(self, block: blocks.StorageBlock, horizon: time.TimeFrame, horizon_index: int) -> None:
        stored_energy = self._optimization_result.get_stored_energy(block, horizon.dti_extd)
        block.states.loc[horizon.dti_extd, "energy"] = stored_energy
        block.states.loc[horizon.dti_extd, "soc"] = (stored_energy / block.sizes["storage"].total).fillna(0)

        # This is a small hack to make the battery aging model compatible with the new results extraction API.
        # TODO: decouple the battery aging model from the horizon and remove this hack.
        comp_horizon = types.SimpleNamespace(ch=horizon, index=horizon_index)
        block.aging_model.age(comp_horizon)


@dataclass
class SimulationSettings:
    solver: optimization.Solver = optimization.Solver.GUROBI
    optimality_tol: float | None = 1e-9
    backend: optimization.OptimizationBackend = optimization.OptimizationBackend.OEMOF
    largescalemode: bool = False
    n_processes: int = 1
    debugmode: bool = False
    rerun_infeasible: bool = True
    key_solcast_api: str | None = None


class PredictionHorizon:
    def __init__(self, index: int, scenario: scn.Scenario, settings: SimulationSettings, logger: logging.Logger):
        self.index = index
        self.scenario = scenario
        self._settings = settings

        # set up the logger as a child of the scenario logger with some additional horizon index metadata
        self._logger = logger_fcs.ContextLoggerAdapter(
            logger=logger, extra={"n_horizon": self.index + 1, "n_horizon_total": self.scenario.nhorizons}
        )

        # region time and data generation and slicing
        start = self.scenario.times.sim.start + (self.index * self.scenario.len_ch)
        self.ph = time.TimeFrame.create_from_start_timestamp(
            start=start,
            timestep=self.scenario.timestep,
            timezone=self.scenario.location.timezone,
            end=min(start + self.scenario.len_ph, self.scenario.times.sim.end),
            start_ref=self.scenario.times.sim.start,
        )

        self.ch = time.TimeFrame.create_from_start_timestamp(
            start=start,
            timestep=self.scenario.timestep,
            timezone=self.scenario.location.timezone,
            end=min(start + self.scenario.len_ch, self.scenario.times.eval.end),
            start_ref=self.scenario.times.sim.start,
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
        Perform the concrete optimization across a prediction horizon.
        """
        self._logger.info("Building optimization problem")
        optimization_problem_config = optimization.OptimizationProblemConfig(
            cost_eps=self.scenario.cost_eps,
            storage_reward_eps=self.scenario.storage_reward_eps,
            optimality_tol=self._settings.optimality_tol,
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

        self._logger.info(f"Optimization problem built; starting optimization with {self._settings.solver}")
        status, optimization_result = optimization_problem.solve()

        if status == optimization.OptimizationStatus.INFEASIBLE_OR_UNBOUNDED:
            raise OptimizationError(
                "Scenario failed: Infeasible or Unbounded (To solve this error try to "
                "set investment limits for blocks or for the scenario)",
                prediction_horizon_idx=self.index,
                prediction_horizon_num=self.scenario.nhorizons,
            )
        elif status != optimization.OptimizationStatus.OPTIMAL:
            raise OptimizationError(
                f"Scenario failed: {status}. Enable debug mode for more information.",
                prediction_horizon_idx=self.index,
                prediction_horizon_num=self.scenario.nhorizons,
            )

        if optimization_result is None:
            raise OptimizationError(
                "Optimization failed: No optimization result, even though the solver signaled an optimal result. This is a bug.",
                prediction_horizon_idx=self.index,
                prediction_horizon_num=self.scenario.nhorizons,
            )

        if self.scenario.nhorizons == 1:  # Don't store objective for multiple horizons in scenario (most RH scenarios)
            self.scenario.objective_opt = optimization_result.get_objective()

        _OptimizationHorizonResultProcessor.collect_optimization_results(
            optimization_result, self.scenario, horizon=self.ch, horizon_index=self.index
        )
