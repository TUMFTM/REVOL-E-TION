import logging
import pprint
from functools import singledispatchmethod
from typing import Any

import oemof.solph as solph
import pandas as pd
import pyomo.environ as po
from typing_extensions import Self, override

import revoletion.optimization.optimization_problem as optimization_problem
from revoletion import blocks, utils
from revoletion import scenario as scn

from ._oemof_block_visitor import OemofBlockVisitor, WrappedEnergySystem


class OemofOptimizationResult(optimization_problem.OptimizationResult):
    _components: WrappedEnergySystem
    _raw_results: dict[Any, Any]
    _objective: float

    def __init__(
        self, components: WrappedEnergySystem, raw_optimization_results: dict[Any, Any], objective: float
    ) -> None:
        super().__init__()
        self._components = components
        self._raw_results = raw_optimization_results
        self._objective = objective

    @singledispatchmethod
    @override
    def get_power_flow(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        """
        Get the power flow time series for a given block.

        :param block: The block for which power flows should be returned.
        :param dti: The datetime index specifying the time horizon of interest.

        :returns: Dictionary mapping flow direction identifiers to power flow series.
        """
        return {}

    @get_power_flow.register
    def _(self, block: blocks.SystemCore, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {
            "acdc": self._get_flow_for_components(block, ("ac", "acdc"), dti),
            "dcac": self._get_flow_for_components(block, ("dc", "dcac"), dti),
        }

    @get_power_flow.register
    def _(self, block: blocks.RenewableSource, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {
            "out": self._get_flow_for_components(block, ("outflow", "bus-connected"), dti),
            "pot": self._get_flow_for_components(block, ("src", "bus"), dti),
            "curt": self._get_flow_for_components(block, ("bus", "exc"), dti),
        }

    @get_power_flow.register
    def _(self, block: blocks.FixedDemand, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {
            "in": self._get_flow_for_components(block, ("bus-connected", "snk"), dti),
        }

    @get_power_flow.register
    def _(self, block: blocks.ControllableSource, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {
            "out": self._get_flow_for_components(block, ("src", "bus-connected"), dti),
        }

    @get_power_flow.register
    def _(self, block: blocks.GridConnection, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        grid_components = self._components.get_components(block)
        bus_component = self._components.get_component(block, "bus")

        flows = {}
        for label, component in grid_components.items():
            if label.startswith("inflow"):
                inflow = self._raw_results[(component, bus_component)]["sequences"]["flow"][dti]
                flows[label] = inflow
            elif label.startswith("outflow"):
                outflow = self._raw_results[(bus_component, component)]["sequences"]["flow"][dti]
                flows[label] = outflow

        return flows

    @get_power_flow.register
    def _(self, block: blocks.GridMarket, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {
            "in": self._get_flow_for_components(block, ("bus-connected", "snk"), dti),
            "out": self._get_flow_for_components(block, ("src", "bus-connected"), dti),
        }

    @get_power_flow.register
    def _get_power_flow_storage(self, block: blocks.StorageBlock, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {
            "out": self._get_flow_for_components(block, ("outflow", "bus-connected"), dti),
            "in": self._get_flow_for_components(block, ("bus-connected", "inflow"), dti),
            "bat_out": self._get_flow_for_components(block, ("storage", "bus"), dti),
            "bat_in": self._get_flow_for_components(block, ("bus", "storage"), dti),
        }

    @get_power_flow.register
    def _(self, block: blocks.Fleet, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {
            "in": self._get_flow_for_components(block, ("outflow", "bus-connected"), dti),
            "out": self._get_flow_for_components(block, ("bus-connected", "inflow"), dti),
        }

    @get_power_flow.register
    def _(self, block: blocks.ElectricFleetUnit, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        """
        Adds external AC/DC conversion flows for fleet units.
        """
        power_flows = self._get_power_flow_storage(block, dti)
        power_flows.update(
            {
                "ext_dc": self._get_flow_for_components(block, ("bus_ext_ac", "conv_ext_ac"), dti),
                "ext_ac": self._get_flow_for_components(block, ("bus_ext_dc", "conv_ext_dc"), dti),
            }
        )
        return power_flows

    def _get_flow_for_components(
        self, block: blocks.ElectricBlock, components: tuple[str, ...], dti: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Internal helper to retrieve oemof flow series for a component pair.

        Components can be referenced by their block scope label and will be automatically resolved.
        """
        remapped_components_key = tuple(map(lambda label: self._components.get_component(block, label), components))
        return self._raw_results[remapped_components_key]["sequences"]["flow"][dti]  # type: ignore

    @override
    def get_stored_energy(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> float | pd.Series:
        """
        Returns stored energy content of storage components.
        """
        energy = self._raw_results[(self._components.get_component(block, "storage"), None)]["sequences"][
            "storage_content"
        ][dti]
        return energy

    @singledispatchmethod
    @override
    def get_opex(self, block: blocks.ElectricBlock) -> float:
        raise NotImplementedError("Operational expenditures are currently not implemented for oemof.")

    @singledispatchmethod
    @override
    def get_expansion(self, block: blocks.ElectricBlock) -> dict[str, float]:
        """
        Returns capital expenditures for a block.

        :returns: Mapping of component identifier to CAPEX.
        """
        return {}

    @get_expansion.register
    def _(self, block: blocks.SystemCore) -> dict[str, float]:
        return {
            "acdc": self._get_expansion_for_components(block, ("ac", "acdc")),
            "dcac": self._get_expansion_for_components(block, ("dc", "dcac")),
        }

    @get_expansion.register
    def _(self, block: blocks.StorageBlock) -> dict[str, float]:
        return {"storage": self._get_expansion_for_components(block, ("storage", None))}

    @get_expansion.register
    def _(self, block: blocks.RenewableSource) -> dict[str, float]:
        return {"block": self._get_expansion_for_components(block, ("src", "bus"))}

    @get_expansion.register
    def _(self, block: blocks.ControllableSource) -> dict[str, float]:
        return {"block": self._get_expansion_for_components(block, ("src", "bus-connected"))}

    @get_expansion.register
    def _(self, block: blocks.GridConnection) -> dict[str, float]:
        grid_components = self._components.get_components(block)

        outflow_components = []
        inflow_components = []
        for label, component in grid_components.items():
            if label.startswith("outflow"):
                outflow_components.append(component)
            elif label.startswith("inflow"):
                inflow_components.append(component)

        bus_component = self._components.get_component(block, "bus")
        g2s = self._raw_results[(bus_component, outflow_components[0])]["scalars"]["invest"]
        s2g = self._raw_results[(inflow_components[0], bus_component)]["scalars"]["invest"]
        return {"g2s": g2s, "s2g": s2g}

    @get_expansion.register
    def _(self, block: blocks.StorageBlock) -> dict[str, float]:
        return {"storage": self._get_expansion_for_components(block, ("storage", None))}

    def _get_expansion_for_components(self, block: blocks.BaseBlock, components: tuple[str | None, ...]) -> float:
        remapped_components_key = tuple(
            self._components.get_component(block, component) if component else None for component in components
        )
        return self._raw_results[remapped_components_key]["scalars"]["invest"]

    @override
    def get_objective(self) -> float:
        return self._objective


VALID_OEMOF_SOLVERS = {optimization_problem.Solver.CBC, optimization_problem.Solver.GUROBI}


class OemofOptimizationProblem(optimization_problem.OptimizationProblem):
    def __init__(
        self,
        energy_system: WrappedEnergySystem,
        scenario: scn.Scenario,
        logger: logging.Logger,
        config: optimization_problem.OptimizationProblemConfig | None = None,
    ) -> None:
        super().__init__(logger, config)
        self._energy_system = energy_system
        self._scenario = scenario

    @override
    @classmethod
    def from_revoletion_scenario(
        cls,
        scenario: scn.Scenario,
        horizon: utils.TimeSettings,
        logger: logging.Logger,
        config: optimization_problem.OptimizationProblemConfig | None = None,
    ) -> Self:
        if config is None:
            config = optimization_problem.OptimizationProblemConfig()

        energy_system = OemofBlockVisitor.create_oemof_energy_system(scenario, horizon, config.cost_eps)

        return cls(energy_system, scenario, logger, config)

    @override
    def set_input_power_unit(self, block: blocks.ElectricBlock, power_unit: float, dti: pd.DatetimeIndex) -> None:
        raise NotImplementedError()

    @override
    def set_output_power_unit(self, block: blocks.ElectricBlock, power_unit: float, dti: pd.DatetimeIndex) -> None:
        raise NotImplementedError()

    @override
    def solve(self) -> tuple[optimization_problem.OptimizationStatus, optimization_problem.OptimizationResult | None]:
        if self._config.solver not in VALID_OEMOF_SOLVERS:
            raise ValueError(
                f"Failed to optimize oemof energy system: solver {self._config.solver.name} is not supported for oemof"
            )

        self._logger.info("Building oemof model")
        model = self._create_model()

        self._logger.info("Model built, starting optimization")
        results = model.solve(
            solver=self._config.solver.value,
            solve_kwargs={"tee": self._config.debug},
            # We explicitly handle the return code, so oemof should not raise an error if the result is not optimal.
            allow_nonoptimal=True,
        )

        # The optimization result has two status codes: one for the solver and one for the termination condition.
        # We only care whether the problem was solved or not, so it is reduced down to a single `OptimizationStatus`.
        status = self._get_optimization_status_from_optimization_results(results)
        if status != optimization_problem.OptimizationStatus.OPTIMAL:
            # No optimization result available, so no further post-processing should be applied.
            return status, None

        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(pprint.pformat(solph.processing.meta_results(model)))

        results_dict = solph.processing.results(model)

        objective = model.objective()
        return status, OemofOptimizationResult(
            components=self._energy_system, raw_optimization_results=results_dict, objective=objective
        )

    @override
    def solve_time_step(
        self, time_step: pd.DatetimeIndex
    ) -> tuple[optimization_problem.OptimizationStatus, optimization_problem.OptimizationResult | None]:
        raise NotImplementedError("Optimizing a single time step is currently not supported with oemof")

    def _create_model(self) -> solph.Model:
        self._logger.info("Building optimization problem from oemof model")

        model = solph.Model(self._energy_system.es, debug=self._config.debug)

        self._energy_system.constraints.apply_constraints(model=model)

        if self._config.debug:
            model.write(self._scenario.paths.dump, format="lp", io_options={"symbolic_solver_labels": True})

        return model  # type: ignore

    def _get_optimization_status_from_optimization_results(self, results) -> optimization_problem.OptimizationStatus:
        if (results.solver.status == po.SolverStatus.ok) and (
            results.solver.termination_condition == po.TerminationCondition.optimal
        ):
            return optimization_problem.OptimizationStatus.OPTIMAL
        elif results.solver.termination_condition == po.TerminationCondition.infeasible:
            return optimization_problem.OptimizationStatus.INFEASIBLE
        elif results.solver.termination_condition == po.TerminationCondition.unbounded:
            return optimization_problem.OptimizationStatus.UNBOUNDED
        elif results.solver.termination_condition == po.TerminationCondition.infeasibleOrUnbounded:
            return optimization_problem.OptimizationStatus.INFEASIBLE_OR_UNBOUNDED
        else:
            return optimization_problem.OptimizationStatus.ERROR
