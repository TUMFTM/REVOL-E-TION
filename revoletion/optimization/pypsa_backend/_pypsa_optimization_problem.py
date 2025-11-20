import contextlib
import logging
import os
import sys
from functools import singledispatchmethod

import linopy.constants
import numpy as np
import pandas as pd
import pypsa
from typing_extensions import Self, override

import revoletion.optimization.optimization_problem as optimization_problem
from revoletion import blocks, utils
from revoletion import scenario as scn

from ._pypsa_block_visitor import PyPSABlockVisitor, make_pypsa_label
from ._utils import normalize_datetime_index

_LOGGER = logging.getLogger(__name__)

_CHARGE_POWER_BUFFER = 1e-6


class PypsaOptimizationResult(optimization_problem.OptimizationResult):
    """
    Result of optimizing a `PyPSAOptimizationProblem`.

    Since PyPSA saves the results directly on the network, this result object is basically just a wrapper around a PyPSA network.
    """

    def __init__(self, net: pypsa.Network) -> None:
        super().__init__()
        self._net = net

    @singledispatchmethod
    @override
    def get_power_flow(self, block: blocks.ElectricBlock, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {}

    @get_power_flow.register
    def _(self, block: blocks.SystemCore, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {
            "acdc": self._get_power_flow_link(block, dti, "acdc-link"),
            "dcac": self._get_power_flow_link(block, dti, "dcac-link"),
        }

    @get_power_flow.register
    def _(self, block: blocks.FixedDemand, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        normalized_dti = normalize_datetime_index(dti)
        pypsa_load_name = make_pypsa_label(block, "load")
        pypsa_load = self._net.loads_t.p.loc[normalized_dti, pypsa_load_name]
        return {"in": self._align_pypsa_values_to_dti(pypsa_load, dti)}

    @get_power_flow.register
    def _(self, block: blocks.RenewableSource, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        # `out` is the
        out = self._get_power_flow_link(block, dti, "outflow-link")

        # `pot` is the potential available power from the generator, which might not be fully utilized.
        normalized_dti = normalize_datetime_index(dti)
        pypsa_gen_name = make_pypsa_label(block, "gen")
        pypsa_pot = self._net.generators_t.p_max_pu.loc[normalized_dti, pypsa_gen_name] * block.sizes["block"].total
        pot = self._align_pypsa_values_to_dti(pypsa_pot, dti)

        # `curt` represents the curtailed power, i.e., the unused amount of available generation power.
        curt = pot - out
        return {"out": out, "pot": pot, "curt": curt}

    @get_power_flow.register
    def _(self, block: blocks.ControllableSource, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {"out": self._get_power_flow_generator(block, dti, "gen")}

    @get_power_flow.register
    def _(self, block: blocks.GridConnection, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        flows: dict[str, pd.Series] = {}

        flows["inflow_1"] = self._get_power_flow_link(block, dti, "inflow-link")

        # This fakes the peak shaving API of oemof to make the API compatible.
        # TODO: implement the peak shaving behavior for PyPSA.
        for period in block.peak_periods.index:
            label = f"outflow_{period}"
            flows[label] = self._get_power_flow_link(
                block, dti, "outflow-link"
            ).copy()  # identical data for each outflow_n

        return flows

    @get_power_flow.register
    def _(self, block: blocks.GridMarket, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {
            "out": self._get_power_flow_generator(block, dti, "export-gen"),
            "in": self._get_power_flow_generator(block, dti, "import-gen"),
        }

    @get_power_flow.register
    def _get_power_flow_storage(self, block: blocks.StorageBlock, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        normalized_dti = normalize_datetime_index(dti)
        pypsa_store_name = make_pypsa_label(block, "battery-store")
        pypsa_bat_power = self._net.stores_t.p.loc[normalized_dti, pypsa_store_name]
        bat_power = self._align_pypsa_values_to_dti(pypsa_bat_power, dti)

        return {
            "out": self._get_power_flow_link(block, dti, "outflow-link"),
            "in": self._get_power_flow_link(block, dti, "inflow-link"),
            "bat_out": bat_power.clip(lower=0),
            "bat_in": -bat_power.clip(upper=0),
        }

    @get_power_flow.register
    def _(self, block: blocks.Fleet, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        return {
            "in": self._get_power_flow_link(block, dti, "inflow-link"),
            "out": self._get_power_flow_link(block, dti, "outflow-link"),
        }

    @get_power_flow.register
    def _(self, block: blocks.ElectricFleetUnit, dti: pd.DatetimeIndex) -> dict[str, pd.Series]:
        power_flows = self._get_power_flow_storage(block, dti)
        power_flows.update(
            {
                "ext_dc": self._get_power_flow_link(block, dti, "ext-dc-inflow-link"),
                "ext_ac": self._get_power_flow_link(block, dti, "ext-ac-inflow-link"),
            }
        )
        return power_flows

    def _get_power_flow_link(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex, label: str) -> pd.Series:
        """
        Helper method to determine the power flow of a PyPSA link.
        """
        normalized_dti = normalize_datetime_index(dti)
        pypsa_link_name = make_pypsa_label(block, label)
        pypsa_power_flow = self._net.links_t.p0.loc[normalized_dti, pypsa_link_name]

        return self._align_pypsa_values_to_dti(pypsa_power_flow, dti)

    def _get_power_flow_generator(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex, label: str) -> pd.Series:
        """
        Helper method to determine the power flow of a PyPSA generator.
        """
        normalized_dti = normalize_datetime_index(dti)
        pypsa_gen_name = make_pypsa_label(block, label)
        pypsa_power_flow = self._net.generators_t.p.loc[normalized_dti, pypsa_gen_name]
        return self._align_pypsa_values_to_dti(pypsa_power_flow, dti)

    def _align_pypsa_values_to_dti(
        self, pypsa_values: pd.Series | pd.DataFrame | float, dti: pd.DatetimeIndex
    ) -> pd.Series | pd.DataFrame | float:
        if isinstance(pypsa_values, float):
            return pypsa_values
        return pypsa_values.tz_localize(dti.tz)

    @override
    def get_stored_energy(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> float | pd.Series:
        if not isinstance(block, blocks.ElectricFleetUnit):
            raise ValueError(f"Cannot determine SoC for block {block.name} of type {type(block)}")

        pypsa_store_name = make_pypsa_label(block, "battery-store")
        if pypsa_store_name in self._net.stores_t.e:
            normalized_dti = normalize_datetime_index(dti)
            pypsa_e_curr = self._net.stores_t.e.loc[normalized_dti, pypsa_store_name]
            e_curr = self._align_pypsa_values_to_dti(pypsa_e_curr, dti)
        else:
            e_curr = self._net.stores.loc[pypsa_store_name, "e_initial"]
        return e_curr

    @singledispatchmethod
    @override
    def get_opex(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> float:
        return 0.0

    @get_opex.register
    def _(self, block: blocks.GridConnection, dti: pd.DatetimeIndex) -> float:
        normalized_dti = normalize_datetime_index(dti)
        pypsa_inflow_link_name = make_pypsa_label(block, "inflow-link")

        grid_conn_inflow_power_wh = self._net.links_t.p0.loc[normalized_dti, pypsa_inflow_link_name]
        grid_conn_inflow_cost = self._net.c.links.static.marginal_cost[pypsa_inflow_link_name]

        grid_market_revenue = [self.get_opex(grid_market_block, dti) for grid_market_block in block.subblocks.values()]

        return grid_conn_inflow_power_wh * grid_conn_inflow_cost + sum(grid_market_revenue)

    @get_opex.register
    def _(self, block: blocks.GridMarket, dti: pd.DatetimeIndex) -> float:
        normalized_dti = normalize_datetime_index(dti)
        pypsa_import_gen_name = make_pypsa_label(block, "import-gen")
        pypsa_export_gen_name = make_pypsa_label(block, "export-gen")

        grid_import_power_wh = self._net.generators_t.p.loc[normalized_dti, pypsa_import_gen_name]
        grid_export_power_wh = self._net.generators_t.p.loc[normalized_dti, pypsa_export_gen_name]

        grid_import_cost = self._net.c.generators.dynamic.marginal_cost.loc[normalized_dti, pypsa_import_gen_name]
        grid_export_cost = self._net.c.generators.dynamic.marginal_cost.loc[normalized_dti, pypsa_export_gen_name]

        return grid_import_power_wh * grid_import_cost + grid_export_power_wh * grid_export_cost

    @get_opex.register
    def _(self, block: blocks.RenewableSource, dti: pd.DatetimeIndex) -> float:
        normalized_dti = normalize_datetime_index(dti)
        pypsa_gen_name = make_pypsa_label(block, "gen")

        power_wh = self._net.generators_t.p.loc[normalized_dti, pypsa_gen_name]
        costs = self._net.c.generators.dynamic.marginal_cost.loc[normalized_dti, pypsa_gen_name]

        return power_wh * costs

    @get_opex.register
    def _(self, block: blocks.ControllableSource, dti: pd.DatetimeIndex) -> float:
        normalized_dti = normalize_datetime_index(dti)
        pypsa_gen_name = make_pypsa_label(block, "gen")

        power_wh = self._net.generators_t.p.loc[normalized_dti, pypsa_gen_name]
        costs = self._net.c.generators.dynamic.marginal_cost.loc[normalized_dti, pypsa_gen_name]

        return power_wh * costs

    @get_opex.register
    def _(self, block: blocks.StationaryBattery, dti: pd.DatetimeIndex) -> float:
        normalized_dti = normalize_datetime_index(dti)
        pypsa_inflow_link_name = make_pypsa_label(block, "inflow-link")
        pypsa_outflow_link_name = make_pypsa_label(block, "outflow-link")

        inflow_power_wh = self._net.links_t.p0.loc[normalized_dti, pypsa_inflow_link_name]
        outflow_power_wh = self._net.links_t.p0.loc[normalized_dti, pypsa_outflow_link_name]

        if pypsa_inflow_link_name in self._net.c.links.dynamic.marginal_cost:
            inflow_costs = self._net.c.links.dynamic.marginal_cost.loc[normalized_dti, pypsa_inflow_link_name]
        else:
            inflow_costs = self._net.c.links.static.marginal_cost[pypsa_inflow_link_name]

        if pypsa_outflow_link_name in self._net.c.links.dynamic.marginal_cost:
            outflow_costs = self._net.c.links.dynamic.marginal_cost.loc[normalized_dti, pypsa_outflow_link_name]
        else:
            outflow_costs = self._net.c.links.static.marginal_cost[pypsa_outflow_link_name]

        return inflow_power_wh * inflow_costs + outflow_power_wh * outflow_costs

    @get_opex.register
    def _(self, block: blocks.ElectricFleetUnit, dti: pd.DatetimeIndex) -> float:
        normalized_dti = normalize_datetime_index(dti)
        pypsa_inflow_link_name = make_pypsa_label(block, "inflow-link")
        pypsa_outflow_link_name = make_pypsa_label(block, "outflow-link")

        inflow_power_wh = self._net.links_t.p0.loc[normalized_dti, pypsa_inflow_link_name]
        outflow_power_wh = self._net.links_t.p0.loc[normalized_dti, pypsa_outflow_link_name]

        inflow_power_wh = self._net.links_t.p0.loc[normalized_dti, pypsa_inflow_link_name]
        outflow_power_wh = self._net.links_t.p0.loc[normalized_dti, pypsa_outflow_link_name]

        if pypsa_inflow_link_name in self._net.c.links.dynamic.marginal_cost:
            inflow_costs = self._net.c.links.dynamic.marginal_cost.loc[normalized_dti, pypsa_inflow_link_name]
        else:
            inflow_costs = self._net.c.links.static.marginal_cost[pypsa_inflow_link_name]

        if pypsa_outflow_link_name in self._net.c.links.dynamic.marginal_cost:
            outflow_costs = self._net.c.links.dynamic.marginal_cost.loc[normalized_dti, pypsa_outflow_link_name]
        else:
            outflow_costs = self._net.c.links.static.marginal_cost[pypsa_outflow_link_name]

        return inflow_power_wh * inflow_costs + outflow_power_wh * outflow_costs

    @singledispatchmethod
    @override
    def get_expansion(self, block: blocks.ElectricBlock) -> dict[str, float]:
        return {}

    @get_expansion.register
    def _(self, block: blocks.SystemCore) -> dict[str, float]:
        return {
            "acdc": self._get_expansion_of_link(block, "acdc-link"),
            "dcac": self._get_expansion_of_link(block, "dcac-link"),
        }

    @get_expansion.register
    def _(self, block: blocks.StationaryBattery) -> dict[str, float]:
        return {"storage": self._get_expansion_of_store(block, "battery-store")}

    @get_expansion.register
    def _(self, block: blocks.RenewableSource) -> dict[str, float]:
        return {"block": self._get_expansion_of_generator(block, "gen")}

    @get_expansion.register
    def _(self, block: blocks.ControllableSource) -> dict[str, float]:
        return {"block": self._get_expansion_of_generator(block, "gen")}

    @get_expansion.register
    def _(self, block: blocks.GridConnection) -> dict[str, float]:
        return {
            "g2s": self._get_expansion_of_link(block, "outflow-link"),
            "s2g": self._get_expansion_of_link(block, "inflow-link"),
        }

    @get_expansion.register
    def _(self, block: blocks.ElectricFleetUnit) -> dict[str, float]:
        return {"storage": self._get_expansion_of_store(block, "battery-store")}

    def _get_expansion_of_generator(self, block: blocks.BaseBlock, label: str) -> float:
        """
        Helper method to get the expansion of a generator.
        """
        pypsa_gen_name = make_pypsa_label(block, label)
        gen_p_nom = self._net.generators.loc[pypsa_gen_name, "p_nom"]
        gen_p_nom_opt = self._net.generators.loc[pypsa_gen_name, "p_nom_opt"]
        gen_p_nom_expansion = gen_p_nom_opt - gen_p_nom
        return gen_p_nom_expansion

    def _get_expansion_of_link(self, block: blocks.BaseBlock, label: str) -> float:
        pypsa_inflow_link_name = make_pypsa_label(block, label)
        link_p_nom = self._net.links.loc[pypsa_inflow_link_name, "p_nom"]
        link_p_nom_opt = self._net.links.loc[pypsa_inflow_link_name, "p_nom_opt"]
        link_p_nom_expansion = link_p_nom_opt - link_p_nom
        return link_p_nom_expansion

    def _get_expansion_of_store(self, block: blocks.BaseBlock, label: str) -> float:
        pypsa_gen_name = make_pypsa_label(block, label)
        store_e_nom = self._net.stores.loc[pypsa_gen_name, "e_nom"]
        store_e_nom_opt = self._net.stores.loc[pypsa_gen_name, "e_nom_opt"]
        store_e_nom_expansion = store_e_nom_opt - store_e_nom
        return store_e_nom_expansion

    @override
    def get_objective(self) -> float:
        if self._net.objective is None:
            raise RuntimeError("Failed to get objective value of PyPSA optimization result. This is a bug.")

        return self._net.objective


VALID_PYPSA_SOLVERS = {
    optimization_problem.Solver.CBC,
    optimization_problem.Solver.HIGHS,
}


class PypsaOptimizationProblem(optimization_problem.OptimizationProblem):
    def __init__(
        self,
        net: pypsa.Network,
        logger: logging.Logger,
        config: optimization_problem.OptimizationProblemConfig | None = None,
    ) -> None:
        super().__init__(logger, config)
        self._net = net

    @property
    def pypsa_network(self) -> pypsa.Network:
        return self._net

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

        visitor = PyPSABlockVisitor(horizon.dti, config.cost_eps)
        pypsa_network = visitor.create_pypsa_network(scenario.block_registry)

        if not config.invest:
            _pypsa_disable_investment_for_network(pypsa_network)

        return cls(pypsa_network, logger, config)

    @override
    def set_input_power_unit(self, block: blocks.ElectricBlock, power_unit: float, dti: pd.DatetimeIndex) -> None:
        if not isinstance(block, blocks.ElectricFleetUnit):
            raise ValueError(f"Cannot set output power unit for block {block.name} of type {type(block)}")

        normalized_dti = normalize_datetime_index(dti)

        charger_in = make_pypsa_label(block, "inflow-link")
        charger_out = make_pypsa_label(block, "outflow-link")

        prev_p_max_pu = self._net.c.links.dynamic.p_max_pu.loc[normalized_dti, charger_out]
        prev_p_min_pu = self._net.c.links.dynamic.p_min_pu.loc[normalized_dti, charger_out]

        p_max_pu = min(power_unit + _CHARGE_POWER_BUFFER, prev_p_max_pu)
        p_min_pu = max(power_unit - _CHARGE_POWER_BUFFER, prev_p_min_pu)

        self._net.c.links.dynamic.p_max_pu.loc[normalized_dti, charger_in] = p_max_pu
        self._net.c.links.dynamic.p_min_pu.loc[normalized_dti, charger_in] = p_min_pu
        # Set the fixed flow to NaN to avoid any numerical issues with the solver and let PyPSA figure out the exact flow.
        self._net.c.links.dynamic.p_set.loc[normalized_dti, charger_in] = np.nan

        # Set the fixed flow to NaN to avoid any numerical issues with the solver and let PyPSA figure out the exact flow.
        self._net.c.links.dynamic.p_set.loc[normalized_dti, charger_out] = np.nan
        self._net.c.links.dynamic.p_max_pu.loc[normalized_dti, charger_out] = 0.0

    @override
    def set_output_power_unit(self, block: blocks.ElectricBlock, power_unit: float, dti: pd.DatetimeIndex) -> None:
        if not isinstance(block, blocks.ElectricFleetUnit):
            raise ValueError(f"Cannot set output power for block {block.name} of type {type(block)}")

        normalized_dti = normalize_datetime_index(dti)

        charger_in = make_pypsa_label(block, "inflow-link")
        charger_out = make_pypsa_label(block, "outflow-link")

        prev_p_max_pu = self._net.c.links.dynamic.p_max_pu.loc[normalized_dti, charger_out]
        prev_p_min_pu = self._net.c.links.dynamic.p_min_pu.loc[normalized_dti, charger_out]

        p_max_pu = min(power_unit + _CHARGE_POWER_BUFFER, prev_p_max_pu)
        p_min_pu = max(power_unit - _CHARGE_POWER_BUFFER, prev_p_min_pu)

        self._net.c.links.dynamic.p_max_pu.loc[normalized_dti, charger_out] = p_max_pu
        self._net.c.links.dynamic.p_min_pu.loc[normalized_dti, charger_out] = p_min_pu
        # Set the fixed flow to NaN to avoid any numerical issues with the solver and let PyPSA figure out the exact flow.
        self._net.c.links.dynamic.p_set.loc[normalized_dti, charger_out] = np.nan

        # Set the fixed flow to NaN to avoid any numerical issues with the solver and let PyPSA figure out the exact flow.
        self._net.c.links.dynamic.p_set.loc[normalized_dti, charger_in] = np.nan
        self._net.c.links.dynamic.p_max_pu.loc[normalized_dti, charger_in] = 0.0

    @override
    def solve(self) -> tuple[optimization_problem.OptimizationStatus, optimization_problem.OptimizationResult]:
        dti = self._net.snapshots

        status = self._do_solve(dti)

        # PyPSA does not provide separate results like OMEOF, and instead the results are directly saved inside the network.
        return status, PypsaOptimizationResult(net=self._net.copy())

    @override
    def solve_time_step(
        self, time_step: pd.DatetimeIndex
    ) -> tuple[optimization_problem.OptimizationStatus, optimization_problem.OptimizationResult]:
        normalized_time_step = normalize_datetime_index(time_step)

        # If we optimize only one time step, we assume rolling horizon optimization.
        # For this case, we must ensure that the initial SoCs are always updated to the previous energy result.
        # Otherwise, the energies would be reset to the canonical initial SoC, which was set during network construction.
        index_loc = self._net.snapshots.get_loc(normalized_time_step)
        # The initial SoCs must not be updated for the first time step.
        if index_loc > 0:
            prev_time_step = self._net.snapshots[index_loc - 1]
            # Only process stores if there are any.
            if not self._net.stores.empty:
                self._net.stores.e_initial = self._net.stores_t.e.loc[prev_time_step]

            # Only process storage units if there are any.
            if not self._net.storage_units.empty:
                self._net.storage_units.state_of_charge_initial = self._net.storage_units_t.state_of_charge.loc[
                    prev_time_step
                ]

        status = self._do_solve(normalized_time_step)

        # PyPSA does not provide separate results like OMEOF, and instead the results are directly saved inside the network.
        # Therefore, the network is just wrapped inside the `PypsaOptimizationResult`.
        return status, PypsaOptimizationResult(net=self._net.copy())

    def _do_solve(self, dti: pd.DatetimeIndex) -> optimization_problem.OptimizationStatus:
        # Try to directly communicate the optimization problem to the HiGHS solver, without writing it to a file.
        # This should reduce the I/O interactions and significantly speed up optimizations for large scenarios.
        io_api = "direct" if self._config.solver == optimization_problem.Solver.HIGHS else None

        with suppress_output():
            solver_status_str, termination_condition_str = self._net.optimize(
                dti,
                # By default, PyPSA and linopy would print status information about the optimization problem to the console.
                # This is quite spammy and therefore it is only enabled for debug mode.
                log_to_console=self._config.debug,
                # Do not show a progress indicator, regardless of debug mode.
                progress=False,
                io_api=io_api,
                solver_name=self._config.solver.value,
                solver_options={
                    "output_flag": False,
                },
            )
        self._logger.debug(
            f"Optimization with solver {self._config.solver.value} finished with status '{solver_status_str}' and termination condition '{termination_condition_str}'"
        )
        solver_status = linopy.constants.SolverStatus(solver_status_str)
        termination_condition = linopy.constants.TerminationCondition(termination_condition_str)
        status = _linopy_status_and_termination_condition_to_optimization_status(solver_status, termination_condition)

        # Cleanup the solver model, to reduce the size of the pypsa network and allow downstream code to copy optimization results.
        self._net.model.solver_model = None

        return status


def _pypsa_disable_investment_for_network(net: pypsa.Network) -> None:
    net.shapes["p_nom_extendable"] = False
    net.buses["p_nom_extendable"] = False
    net.lines["p_nom_extendable"] = False
    net.transformers["p_nom_extendable"] = False
    net.links["p_nom_extendable"] = False
    net.loads["p_nom_extendable"] = False
    net.generators["p_nom_extendable"] = False
    net.storage_units["p_nom_extendable"] = False
    net.stores["p_nom_extendable"] = False
    net.shapes["p_nom_extendable"] = False


def _linopy_status_and_termination_condition_to_optimization_status(
    solver_status: linopy.constants.SolverStatus,
    termination_condition: linopy.constants.TerminationCondition,
) -> optimization_problem.OptimizationStatus:
    match solver_status, termination_condition:
        case linopy.constants.SolverStatus.ok, linopy.constants.TerminationCondition.optimal:
            return optimization_problem.OptimizationStatus.OPTIMAL
        case _, linopy.constants.TerminationCondition.unbounded:
            return optimization_problem.OptimizationStatus.UNBOUNDED
        case _, linopy.constants.TerminationCondition.infeasible:
            return optimization_problem.OptimizationStatus.INFEASIBLE
        case _, linopy.constants.TerminationCondition.infeasible_or_unbounded:
            return optimization_problem.OptimizationStatus.INFEASIBLE_OR_UNBOUNDED
        case (
            _,
            linopy.constants.TerminationCondition.error | linopy.constants.TerminationCondition.internal_solver_error,
        ):
            return optimization_problem.OptimizationStatus.ERROR
        case _, _:
            return optimization_problem.OptimizationStatus.OTHER


@contextlib.contextmanager
def suppress_output():
    """Redirect both Python and C-level stdout/stderr to os.devnull."""
    with open(os.devnull, "w") as devnull:
        # Save original file descriptors
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            # Flush any pending text
            sys.stdout.flush()
            sys.stderr.flush()
            # Redirect low-level fds to devnull
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            # Restore fds
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
