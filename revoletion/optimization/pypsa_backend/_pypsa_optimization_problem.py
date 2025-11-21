import contextlib
import logging
import os
import sys
import tempfile
from functools import singledispatchmethod
from pathlib import Path

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
    def get_power_flow(
        self, block: blocks.ElectricBlock, dti: pd.DatetimeIndex
    ) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        return {}

    @get_power_flow.register
    def _(self, block: blocks.SystemCore, dti: pd.DatetimeIndex) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        return {
            "acdc": self._get_pypsa_link_power_flow(block, dti, "acdc-link"),
            "dcac": self._get_pypsa_link_power_flow(block, dti, "dcac-link"),
        }

    @get_power_flow.register
    def _(self, block: blocks.FixedDemand, dti: pd.DatetimeIndex) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        return {"in": self._get_pypsa_link_power_flow(block, dti, "inflow-link")}

    @get_power_flow.register
    def _(
        self, block: blocks.RenewableSource, dti: pd.DatetimeIndex
    ) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        # `out` is the
        out = self._get_pypsa_link_power_flow(block, dti, "outflow-link")

        # `pot` is the potential available power from the generator, which might not be fully utilized.
        normalized_dti = normalize_datetime_index(dti)
        pypsa_gen_name = make_pypsa_label(block, "gen")
        pypsa_pot = self._net.generators_t.p_max_pu.loc[normalized_dti, pypsa_gen_name] * block.sizes["block"].total
        pot = self._align_pypsa_values_to_dti(pypsa_pot, dti)

        # `curt` represents the curtailed power, i.e., the unused amount of available generation power.
        curt = pot - out
        return {"out": out, "pot": pot, "curt": curt}

    @get_power_flow.register
    def _(
        self, block: blocks.ControllableSource, dti: pd.DatetimeIndex
    ) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        return {"out": self._get_pypsa_power_flow_generator(block, dti, "gen")}

    @get_power_flow.register
    def _(
        self, block: blocks.GridConnection, dti: pd.DatetimeIndex
    ) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        flows: dict[str, pd.Series] = {}

        flows["inflow_1"] = self._get_pypsa_link_power_flow(block, dti, "inflow-link")

        # This fakes the peak shaving API of oemof to make the API compatible.
        # TODO: implement the peak shaving behavior for PyPSA.
        for period in block.peak_periods.index:
            label = f"outflow_{period}"
            flows[label] = self._get_pypsa_link_power_flow(
                block, dti, "outflow-link"
            ).copy()  # identical data for each outflow_n

        return flows

    @get_power_flow.register
    def _(self, block: blocks.GridMarket, dti: pd.DatetimeIndex) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        return {
            "out": self._get_pypsa_power_flow_generator(block, dti, "export-gen"),
            "in": self._get_pypsa_power_flow_generator(block, dti, "import-gen"),
        }

    @get_power_flow.register
    def _get_power_flow_storage(
        self, block: blocks.StorageBlock, dti: pd.DatetimeIndex
    ) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        normalized_dti = normalize_datetime_index(dti)
        pypsa_store_name = make_pypsa_label(block, "battery-store")
        pypsa_bat_power = self._net.stores_t.p.loc[normalized_dti, pypsa_store_name]
        bat_power = self._align_pypsa_values_to_dti(pypsa_bat_power, dti)

        return {
            "out": self._get_pypsa_link_power_flow(block, dti, "outflow-link"),
            "in": self._get_pypsa_link_power_flow(block, dti, "inflow-link"),
            "bat_out": bat_power.clip(lower=0),
            "bat_in": -bat_power.clip(upper=0),
        }

    @get_power_flow.register
    def _(self, block: blocks.Fleet, dti: pd.DatetimeIndex) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        return {
            "in": self._get_pypsa_link_power_flow(block, dti, "inflow-link"),
            "out": self._get_pypsa_link_power_flow(block, dti, "outflow-link"),
        }

    @get_power_flow.register
    def _(
        self, block: blocks.ElectricFleetUnit, dti: pd.DatetimeIndex
    ) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        power_flows = self._get_power_flow_storage(block, dti)
        power_flows.update(
            {
                "ext_dc": self._get_pypsa_link_power_flow(block, dti, "ext-dc-inflow-link"),
                "ext_ac": self._get_pypsa_link_power_flow(block, dti, "ext-ac-inflow-link"),
            }
        )
        return power_flows

    def _get_pypsa_link_power_flow(
        self, block: blocks.BaseBlock, dti: pd.DatetimeIndex, label: str
    ) -> optimization_problem.FloatOrTimeSeries:
        """
        Helper method to determine the power flow of a PyPSA link.
        """
        normalized_dti = normalize_datetime_index(dti)
        pypsa_link_name = make_pypsa_label(block, label)
        pypsa_power_flow = self._net.links_t.p0.loc[normalized_dti, pypsa_link_name]

        return self._align_pypsa_values_to_dti(pypsa_power_flow, dti)

    def _get_pypsa_power_flow_generator(
        self, block: blocks.BaseBlock, dti: pd.DatetimeIndex, label: str
    ) -> optimization_problem.FloatOrTimeSeries:
        """
        Helper method to determine the power flow of a PyPSA generator.
        """
        normalized_dti = normalize_datetime_index(dti)
        pypsa_gen_name = make_pypsa_label(block, label)
        pypsa_power_flow = self._net.generators_t.p.loc[normalized_dti, pypsa_gen_name]
        return self._align_pypsa_values_to_dti(pypsa_power_flow, dti)

    def _align_pypsa_values_to_dti(
        self, pypsa_values: optimization_problem.FloatOrTimeSeries, dti: pd.DatetimeIndex
    ) -> optimization_problem.FloatOrTimeSeries:
        if isinstance(pypsa_values, float):
            return pypsa_values
        return pypsa_values.tz_localize(dti.tz)

    @override
    def get_stored_energy(
        self, block: blocks.BaseBlock, dti: pd.DatetimeIndex
    ) -> optimization_problem.FloatOrTimeSeries:
        if not isinstance(block, blocks.ElectricFleetUnit):
            raise ValueError(f"Cannot determine SoC for block {block.name} of type {type(block)}")

        pypsa_store_name = make_pypsa_label(block, "battery-store")
        normalized_dti = normalize_datetime_index(dti)
        pypsa_store_e = self._net.stores_t.e.loc[normalized_dti, pypsa_store_name]
        return self._align_pypsa_values_to_dti(pypsa_store_e, dti)

    @singledispatchmethod
    @override
    def get_opex(self, block: blocks.BaseBlock, dti: pd.DatetimeIndex) -> optimization_problem.FloatOrTimeSeries:
        return 0.0

    @get_opex.register
    def _(self, block: blocks.GridConnection, dti: pd.DatetimeIndex) -> optimization_problem.FloatOrTimeSeries:
        grid_conn_opex = self._get_pypsa_link_opex(block, dti, "inflow-link")

        grid_market_opex = [self.get_opex(grid_market_block, dti) for grid_market_block in block.subblocks.values()]

        return grid_conn_opex + sum(grid_market_opex)

    @get_opex.register
    def _(self, block: blocks.GridMarket, dti: pd.DatetimeIndex) -> optimization_problem.FloatOrTimeSeries:
        grid_import_opex = self._get_pypsa_generator_opex(block, dti, "import-gen")
        grid_export_opex = self._get_pypsa_generator_opex(block, dti, "export-gen")

        return grid_import_opex + grid_export_opex

    @get_opex.register
    def _(self, block: blocks.RenewableSource, dti: pd.DatetimeIndex) -> optimization_problem.FloatOrTimeSeries:
        return self._get_pypsa_generator_opex(block, dti, "gen")

    @get_opex.register
    def _(self, block: blocks.ControllableSource, dti: pd.DatetimeIndex) -> optimization_problem.FloatOrTimeSeries:
        return self._get_pypsa_generator_opex(block, dti, "gen")

    @get_opex.register
    def _(self, block: blocks.StorageBlock, dti: pd.DatetimeIndex) -> float:
        inflow_link_opex = self._get_pypsa_link_opex(block, dti, "inflow-link")
        outflow_link_opex = self._get_pypsa_link_opex(block, dti, "outflow-link")

        return inflow_link_opex + outflow_link_opex

    def _get_pypsa_link_opex(
        self, block: blocks.BaseBlock, dti: pd.DatetimeIndex, label: str
    ) -> optimization_problem.FloatOrTimeSeries:
        normalized_dti = normalize_datetime_index(dti)
        pypsa_link_name = make_pypsa_label(block, label)

        pypsa_power_flow = self._net.links_t.p0.loc[normalized_dti, pypsa_link_name]

        if pypsa_link_name in self._net.c.links.dynamic.marginal_cost:
            marginal_costs = self._net.c.links.dynamic.marginal_cost.loc[normalized_dti, pypsa_link_name]
        else:
            marginal_costs = self._net.c.links.static.marginal_cost[pypsa_link_name]

        opex = pypsa_power_flow * marginal_costs
        return self._align_pypsa_values_to_dti(opex, dti)

    def _get_pypsa_generator_opex(
        self, block: blocks.BaseBlock, dti: pd.DatetimeIndex, label: str
    ) -> optimization_problem.FloatOrTimeSeries:
        normalized_dti = normalize_datetime_index(dti)
        pypsa_gen_name = make_pypsa_label(block, label)

        pypsa_power_flow = self._net.generators_t.p.loc[normalized_dti, pypsa_gen_name]

        if pypsa_gen_name in self._net.c.generators.dynamic.marginal_cost:
            marginal_costs = self._net.c.generators.dynamic.marginal_cost.loc[normalized_dti, pypsa_gen_name]
        else:
            marginal_costs = self._net.c.generators.static.marginal_cost[pypsa_gen_name]

        opex = pypsa_power_flow * marginal_costs
        return self._align_pypsa_values_to_dti(opex, dti)

    @singledispatchmethod
    @override
    def get_expansion(self, block: blocks.ElectricBlock) -> dict[str, float]:
        return {}

    @get_expansion.register
    def _(self, block: blocks.SystemCore) -> dict[str, float]:
        return {
            "acdc": self._get_pypsa_link_expansion(block, "acdc-link"),
            "dcac": self._get_pypsa_link_expansion(block, "dcac-link"),
        }

    @get_expansion.register
    def _(self, block: blocks.StationaryBattery) -> dict[str, float]:
        return {"storage": self._get_pypsa_store_expansion(block, "battery-store")}

    @get_expansion.register
    def _(self, block: blocks.RenewableSource) -> dict[str, float]:
        return {"block": self._get_pypsa_generator_expansion(block, "gen")}

    @get_expansion.register
    def _(self, block: blocks.ControllableSource) -> dict[str, float]:
        return {"block": self._get_pypsa_generator_expansion(block, "gen")}

    @get_expansion.register
    def _(self, block: blocks.GridConnection) -> dict[str, float]:
        return {
            "g2s": self._get_pypsa_link_expansion(block, "outflow-link"),
            "s2g": self._get_pypsa_link_expansion(block, "inflow-link"),
        }

    @get_expansion.register
    def _(self, block: blocks.ElectricFleetUnit) -> dict[str, float]:
        return {"storage": self._get_pypsa_store_expansion(block, "battery-store")}

    def _get_pypsa_generator_expansion(self, block: blocks.BaseBlock, label: str) -> float:
        """
        Helper method to get the expansion of a generator.
        """
        pypsa_gen_name = make_pypsa_label(block, label)
        gen_p_nom = self._net.generators.loc[pypsa_gen_name, "p_nom"]
        gen_p_nom_opt = self._net.generators.loc[pypsa_gen_name, "p_nom_opt"]
        gen_p_nom_expansion = gen_p_nom_opt - gen_p_nom
        return gen_p_nom_expansion

    def _get_pypsa_link_expansion(self, block: blocks.BaseBlock, label: str) -> float:
        pypsa_inflow_link_name = make_pypsa_label(block, label)
        link_p_nom = self._net.links.loc[pypsa_inflow_link_name, "p_nom"]
        link_p_nom_opt = self._net.links.loc[pypsa_inflow_link_name, "p_nom_opt"]
        link_p_nom_expansion = link_p_nom_opt - link_p_nom
        return link_p_nom_expansion

    def _get_pypsa_store_expansion(self, block: blocks.BaseBlock, label: str) -> float:
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
        self._model = self._net.optimize.create_model()

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

        visitor = PyPSABlockVisitor(
            horizon.dti, config.cost_eps, enable_investment=config.invest, enable_fixed_dispatch=False
        )
        pypsa_network = visitor.create_pypsa_network(scenario.block_registry)

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

        inflow_capacity = block.pwr_chg_max

        p_max = p_max_pu * inflow_capacity
        p_min = p_min_pu * inflow_capacity

        self._model.constraints["Link-fix-p-upper"].rhs.loc[normalized_dti, charger_in] = p_max
        self._model.constraints["Link-fix-p-lower"].rhs.loc[normalized_dti, charger_in] = p_min

        # Set the fixed flow to NaN to avoid any numerical issues with the solver and let PyPSA figure out the exact flow.
        self._net.c.links.dynamic.p_set.loc[normalized_dti, charger_in] = np.nan

        # Set the fixed flow to NaN to avoid any numerical issues with the solver and let PyPSA figure out the exact flow.
        self._net.c.links.dynamic.p_set.loc[normalized_dti, charger_out] = np.nan
        self._net.c.links.dynamic.p_max_pu.loc[normalized_dti, charger_out] = 0.0
        self._model.constraints["Link-fix-p-upper"].rhs.loc[normalized_dti, charger_out] = 0.0

    @override
    def set_output_power_unit(self, block: blocks.ElectricBlock, power_unit: float, dti: pd.DatetimeIndex) -> None:
        if not isinstance(block, blocks.ElectricFleetUnit):
            raise ValueError(f"Cannot set output power for block {block.name} of type {type(block)}")

        normalized_dti = normalize_datetime_index(dti)

        charger_in = make_pypsa_label(block, "inflow-link")
        charger_out = make_pypsa_label(block, "outflow-link")

        outflow_capacity = block.pwr_dis_max * block.eff["dis_int"]

        prev_p_max_pu = self._net.c.links.dynamic.p_max_pu.loc[normalized_dti, charger_out]
        prev_p_min_pu = self._net.c.links.dynamic.p_min_pu.loc[normalized_dti, charger_out]

        p_max_pu = min(power_unit + _CHARGE_POWER_BUFFER, prev_p_max_pu)
        p_min_pu = max(power_unit - _CHARGE_POWER_BUFFER, prev_p_min_pu)

        p_max = p_max_pu * outflow_capacity
        p_min = p_min_pu * outflow_capacity

        self._net.c.links.dynamic.p_max_pu.loc[normalized_dti, charger_out] = p_max_pu
        self._net.c.links.dynamic.p_min_pu.loc[normalized_dti, charger_out] = p_min_pu

        outflow_capacity = block.pwr_dis_max * block.eff["dis_int"]
        p_max = p_max_pu * outflow_capacity
        p_min = p_min_pu * outflow_capacity

        self._model.constraints["Link-fix-p-upper"].rhs.loc[normalized_dti, charger_out] = p_max
        self._model.constraints["Link-fix-p-lower"].rhs.loc[normalized_dti, charger_out] = p_min

        # Set the fixed flow to NaN to avoid any numerical issues with the solver and let PyPSA figure out the exact flow.
        self._net.c.links.dynamic.p_set.loc[normalized_dti, charger_out] = np.nan

        # Set the fixed flow to NaN to avoid any numerical issues with the solver and let PyPSA figure out the exact flow.
        self._net.c.links.dynamic.p_set.loc[normalized_dti, charger_in] = np.nan
        self._net.c.links.dynamic.p_max_pu.loc[normalized_dti, charger_in] = 0.0

        self._model.constraints["Link-fix-p-upper"].rhs.loc[normalized_dti, charger_in] = 0.0

    @override
    def solve(self) -> tuple[optimization_problem.OptimizationStatus, optimization_problem.OptimizationResult]:
        dti = self._net.snapshots

        status = self._do_solve(dti)

        # PyPSA does not provide separate results like OMEOF, and instead the results are directly saved inside the network.
        return status, PypsaOptimizationResult(net=self._net)

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
        return status, PypsaOptimizationResult(net=self._net)

    def _do_solve(self, dti: pd.DatetimeIndex) -> optimization_problem.OptimizationStatus:
        # Try to directly communicate the optimization problem to the HiGHS solver, without writing it to a file.
        # This should reduce the I/O interactions and significantly speed up optimizations for large scenarios.
        io_api = "direct" if self._config.solver == optimization_problem.Solver.HIGHS else None

        self._net._model = self._model
        with suppress_output():
            # Solve the created model.
            solver_status_str, termination_condition_str = self._net.optimize.solve_model(
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
        del self._net.model

        self._logger.debug(
            f"Optimization with solver {self._config.solver.value} finished with status '{solver_status_str}' and termination condition '{termination_condition_str}'"
        )
        solver_status = linopy.constants.SolverStatus(solver_status_str)
        termination_condition = linopy.constants.TerminationCondition(termination_condition_str)
        status = _linopy_status_and_termination_condition_to_optimization_status(solver_status, termination_condition)

        return status


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
