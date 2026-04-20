#!/usr/bin/env python3

import logging
import pprint

import oemof.solph as solph
import pyomo.environ as po

from revoletion import scenario as scn

from . import constraints, time
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


class PredictionHorizon:
    def __init__(self, index: int, scenario: scn.Scenario, logger: logging.Logger):
        self.index = index
        self.scenario = scenario

        # set up the logger as a child of the scenario logger with some additional horizon index metadata
        self._logger = logger_fcs.ContextLoggerAdapter(
            logger=logger, extra={"n_horizon": self.index + 1, "n_horizon_total": self.scenario.nhorizons}
        )

        self._results = None

        # region time and data generation and slicing
        start = self.scenario.times.sim.start + (self.index * self.scenario.len_ch)
        self.ph = time.TimeFrame.create_from_start_timestamp(
            start=start,
            timestep=self.scenario.timestep,
            timezone=self.scenario.location.timezone,
            end=min(start + self.scenario.len_ph, self.scenario.times.sim.end),
        )

        self.ch = time.TimeFrame.create_from_start_timestamp(
            start=start,
            timestep=self.scenario.timestep,
            timezone=self.scenario.location.timezone,
            end=min(start + self.scenario.len_ch, self.scenario.times.eval.end),
        )

        # Display logger message if PH exceeds simulation end time and has to be truncated
        if self.ph.duration < self.scenario.len_ph:
            self._logger.info(msg="Prediction Horizon truncated to simulation end time")

        self._logger.info("Start: %s - CH end: %s - PH end: %s", self.ph.start, self.ch.end, self.ph.end)

        self.es = solph.EnergySystem(
            timeindex=self.ph.dti, infer_last_interval=True
        )  # initialize energy system model instance

        self.constraints = constraints.CustomConstraints(scenario=self.scenario)

        # if apriori power scheduling is necessary, calculate power schedules:
        if self.scenario.scheduler:
            self._logger.debug("Calculating power schedules for commodities with rulebased charging strategies")
            self.scenario.scheduler.calc_ph_schedule(self.ph)
        # endregion

    @property
    def results(self):
        """Get the results of the previous execution."""
        return self._results

    def execute(self) -> None:
        """
        Perform the concrete optimization across a prediction horizon.
        """

        self._logger.info("Building oemof model")
        self._pre_horizon()
        self._logger.debug("Model build completed")

        model = self._create_model()

        self._logger.info("Model built, starting optimization")
        results = model.solve(
            solver=self.scenario.settings.solver, solve_kwargs={"tee": self.scenario.settings.debugmode}
        )
        if (results.solver.status == po.SolverStatus.ok) and (
            results.solver.termination_condition == po.TerminationCondition.optimal
        ):
            self._logger.info("Optimization completed, getting results")
            if (
                self.scenario.nhorizons == 1
            ):  # Don't store objective for multiple horizons in scenario (most RH scenarios)
                self.scenario.objective_opt = model.objective()
        elif results.solver.termination_condition == po.TerminationCondition.infeasible:
            raise OptimizationError(
                "Scenario failed: Infeasible",
                prediction_horizon_idx=self.index,
                prediction_horizon_num=self.scenario.nhorizons,
            )
        elif results.solver.termination_condition == po.TerminationCondition.unbounded:
            raise OptimizationError(
                "Scenario failed: Unbounded",
                prediction_horizon_idx=self.index,
                prediction_horizon_num=self.scenario.nhorizons,
            )
        elif results.solver.termination_condition == po.TerminationCondition.infeasibleOrUnbounded:
            raise OptimizationError(
                "Scenario failed: Infeasible or Unbounded (To solve this error try to "
                "set investment limits for blocks or for the scenario)",
                prediction_horizon_idx=self.index,
                prediction_horizon_num=self.scenario.nhorizons,
            )
        else:
            raise Exception(f"Optimization terminated with unknown status: {results.solver.termination_condition}")

        self._logger.debug(pprint.pformat(solph.processing.meta_results(model)))

        # Get result data slice for current CH from results and save in result dataframes for later analysis
        # Get (possibly optimized) component sizes from results to handle outputs more easily
        self._results = solph.processing.results(model)

        self._post_horizon()

    def _create_model(self) -> solph.Model:
        self._logger.info("Building optimization problem from oemof model")

        model = solph.Model(self.es, debug=self.scenario.settings.debugmode)
        self.constraints.apply_constraints(model=model)

        if self.scenario.settings.debugmode and self.index == 1:
            model.write(self.scenario.paths.dump, io_options={"symbolic_solver_labels": True})

        return model

    def _pre_horizon(self) -> None:
        for block in self.scenario.block_registry.get("TopLevelBlock", {}).values():
            block.pre_horizon(self)

    def _post_horizon(self) -> None:
        for block in self.scenario.block_registry.get("TopLevelBlock", {}).values():
            block.post_horizon(self)
