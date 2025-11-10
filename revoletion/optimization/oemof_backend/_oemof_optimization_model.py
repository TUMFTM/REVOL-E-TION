import logging
import pprint
from functools import singledispatchmethod
from typing import Any

import oemof.solph as solph
import pandas as pd
import pyomo.environ as po
from typing_extensions import Self, override

from revoletion import blocks
from revoletion import scenario as scn
from revoletion.optimization import base

from ._oemof_block_visitor import OemofBlockVisitor, WrappedEnergySystem


class OemofOptimizationResult(base.OptimizationResult):
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
        remapped_components_key = tuple(map(lambda label: self._components.get_component(block, label), components))
        return self._raw_results[remapped_components_key]["sequences"]["flow"][dti]  # type: ignore

    @override
    def get_stored_energy(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> float | pd.Series:
        energy = self._raw_results[(self._components.get_component(block, "storage"), None)]["sequences"][
            "storage_content"
        ][dti]
        return energy

    @override
    def get_opex(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> float: ...

    @singledispatchmethod
    @override
    def get_capex(self, block: blocks.ElectricBlock) -> dict[str, float]:
        return {}

    @get_capex.register
    def _(self, block: blocks.SystemCore) -> dict[str, float]:
        return {
            "acdc": self._get_capex_for_components(block, ("ac", "acdc")),
            "dcac": self._get_capex_for_components(block, ("dc", "dcac")),
        }

    @get_capex.register
    def _(self, block: blocks.StorageBlock) -> dict[str, float]:
        return {"storage": self._get_capex_for_components(block, ("storage", None))}

    @get_capex.register
    def _(self, block: blocks.RenewableSource) -> dict[str, float]:
        return {"block": self._get_capex_for_components(block, ("src", "bus"))}

    @get_capex.register
    def _(self, block: blocks.ControllableSource) -> dict[str, float]:
        return {"block": self._get_capex_for_components(block, ("src", "bus-connected"))}

    @get_capex.register
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

    @get_capex.register
    def _(self, block: blocks.StorageBlock) -> dict[str, float]:
        return {"storage": self._get_capex_for_components(block, ("storage", None))}

    def _get_capex_for_components(self, block: blocks.BaseBlock, components: tuple[str | None, ...]) -> float:
        remapped_components_key = tuple(
            self._components.get_component(block, component) if component else None for component in components
        )
        return self._raw_results[remapped_components_key]["scalars"]["invest"]

    @override
    def get_objective(self) -> float:
        return self._objective


_VALID_OEMOF_SOLVERS = {base.Solver.CBC, base.Solver.GUROBI}


class OemofOptimizationModel(base.OptimizationModel):
    def __init__(
        self,
        energy_system: WrappedEnergySystem,
        scenario: scn.Scenario,
        logger: logging.Logger,
        config: base.OptimizationModelConfig | None = None,
    ) -> None:
        super().__init__(logger, config)
        self._energy_system = energy_system
        self._scenario = scenario

    @override
    @classmethod
    def from_revoletion_scenario(
        cls,
        scenario: scn.Scenario,
        horizon: scn.TimeSettings,
        logger: logging.Logger,
        config: base.OptimizationModelConfig | None = None,
    ) -> Self:
        if config is None:
            config = base.OptimizationModelConfig()

        energy_system = OemofBlockVisitor.create_oemof_energy_system(scenario, horizon, config.cost_eps)

        return cls(energy_system, scenario, logger, config)

    @override
    def set_input_power_unit(self, block: blocks.ElectricBlock, power_unit: float, dti: pd.DatetimeIndex) -> None:
        raise NotImplementedError()

    @override
    def set_output_power(self, block: blocks.ElectricBlock, power: float, dti: pd.DatetimeIndex) -> None:
        raise NotImplementedError()

    @override
    def optimize(
        self, dti: pd.DatetimeIndex | None = None
    ) -> tuple[base.OptimizationStatus, base.OptimizationResult | None]:
        if dti is None:
            dti = self._energy_system.es.timeindex

        return self._do_optimize(dti)

    @override
    def optimize_time_step(
        self, time_step: pd.DatetimeIndex
    ) -> tuple[base.OptimizationStatus, base.OptimizationResult | None]:
        raise NotImplementedError("Optimizing a single time step is currently not supported with OEMOF")

    def _do_optimize(self, dti: pd.DatetimeIndex) -> tuple[base.OptimizationStatus, OemofOptimizationResult | None]:
        if self._config.solver not in _VALID_OEMOF_SOLVERS:
            raise ValueError(
                f"Failed to optimize OEMOF energy system: solver {self._config.solver.name} is not supported for OEMOF"
            )

        self._logger.info("Building oemof model")
        model = self._create_model()
        self._logger.debug("Model build completed")

        results = model.solve(
            solver=self._config.solver.value,
            solve_kwargs={"tee": self._config.debug},
            # We explicitly handle the return code, so OEMOF should not raise an error if the result is not optimal.
            allow_nonoptimal=True,
        )

        status = self._get_optimization_status_from_optimization_results(results)
        if status != base.OptimizationStatus.OPTIMAL:
            return status, None

        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(pprint.pformat(solph.processing.meta_results(model)))

        results_dict = solph.processing.results(model)

        objective = model.objective()
        return status, OemofOptimizationResult(
            components=self._energy_system, raw_optimization_results=results_dict, objective=objective
        )

    def _create_model(self) -> solph.Model:
        self._logger.info("Building optimization problem from oemof model")

        model = solph.Model(self._energy_system.es, debug=self._config.debug)

        self._energy_system.constraints.apply_constraints(model=model)

        # if self._config.debug:
        #     model.write(self._scenario.paths.dump, io_options={"symbolic_solver_labels": True})

        return model

    def _get_optimization_status_from_optimization_results(self, results) -> base.OptimizationStatus:
        if (results.solver.status == po.SolverStatus.ok) and (
            results.solver.termination_condition == po.TerminationCondition.optimal
        ):
            return base.OptimizationStatus.OPTIMAL
        elif results.solver.termination_condition == po.TerminationCondition.infeasible:
            return base.OptimizationStatus.INFEASIBLE
        elif results.solver.termination_condition == po.TerminationCondition.unbounded:
            return base.OptimizationStatus.UNBOUNDED
        elif results.solver.termination_condition == po.TerminationCondition.infeasibleOrUnbounded:
            return base.OptimizationStatus.INFEASIBLE_OR_UNBOUNDED
        else:
            return base.OptimizationStatus.ERROR
