import contextlib
import logging
import os
import sys
import tempfile
from functools import singledispatchmethod
from typing import Any

import linopy.constants
import numpy as np
import pandas as pd
import pypsa
from typing_extensions import Self, override

import revoletion.optimization.optimization_problem as optimization_problem
from revoletion import blocks, time
from revoletion import scenario as scn

from . import _utils as pypsa_utils
from ._pypsa_block_visitor import PyPSABlockVisitor, make_pypsa_label

_LOGGER = logging.getLogger(__name__)


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
        out = self._get_pypsa_link_power_flow(block, dti, "outflow-link")

        # `pot` is the potential available power from the generator, which might not be fully utilized.
        normalized_dti = pypsa_utils.normalize_dti_or_df(dti)
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
        return {"out": self._get_pypsa_generator_power_flow(block, dti, "gen")}

    @get_power_flow.register
    def _(
        self, block: blocks.GridConnection, dti: pd.DatetimeIndex
    ) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        return {
            "in": self._get_pypsa_link_power_flow(block, dti, "inflow-link"),
            "out": self._get_pypsa_link_power_flow(block, dti, "outflow-link"),
        }

    @get_power_flow.register
    def _(self, block: blocks.GridMarket, dti: pd.DatetimeIndex) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        return {
            "out": self._get_pypsa_generator_power_flow(block, dti, "export-gen"),
            "in": self._get_pypsa_generator_power_flow(block, dti, "import-gen"),
        }

    @get_power_flow.register
    def _get_power_flow_storage_block(
        self, block: blocks.StorageBlock, dti: pd.DatetimeIndex
    ) -> dict[str, optimization_problem.FloatOrTimeSeries]:
        normalized_dti = pypsa_utils.normalize_dti_or_df(dti)
        pypsa_store_name = make_pypsa_label(block, "battery-store")
        pypsa_bat_power = self._net.stores_t.p.loc[normalized_dti, pypsa_store_name]
        bat_power = self._align_pypsa_values_to_dti(pypsa_bat_power, dti)

        if isinstance(bat_power, (pd.Series, pd.DatetimeIndex)):
            bat_in = -bat_power.clip(upper=0)
            bat_out = bat_power.clip(lower=0)
        else:
            bat_in = -np.clip(bat_power, a_max=0.0, a_min=-np.inf)
            bat_out = np.clip(bat_power, a_min=0.0, a_max=np.inf)

        return {
            "out": self._get_pypsa_link_power_flow(block, dti, "outflow-link"),
            "in": self._get_pypsa_link_power_flow(block, dti, "inflow-link"),
            "bat_out": bat_out,
            "bat_in": bat_in,
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
        power_flows = self._get_power_flow_storage_block(block, dti)
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
        normalized_dti = pypsa_utils.normalize_dti_or_df(dti)
        pypsa_link_name = make_pypsa_label(block, label)
        pypsa_power_flow = self._net.links_t.p0.loc[normalized_dti, pypsa_link_name]

        return self._align_pypsa_values_to_dti(pypsa_power_flow, dti)

    def _get_pypsa_generator_power_flow(
        self, block: blocks.BaseBlock, dti: pd.DatetimeIndex, label: str
    ) -> optimization_problem.FloatOrTimeSeries:
        """
        Helper method to determine the power flow of a PyPSA generator.
        """
        normalized_dti = pypsa_utils.normalize_dti_or_df(dti)
        pypsa_gen_name = make_pypsa_label(block, label)
        pypsa_power_flow = self._net.generators_t.p.loc[normalized_dti, pypsa_gen_name]
        return self._align_pypsa_values_to_dti(pypsa_power_flow, dti)

    def _align_pypsa_values_to_dti(
        self, pypsa_values: optimization_problem.FloatOrTimeSeries, dti: pd.DatetimeIndex
    ) -> optimization_problem.FloatOrTimeSeries:
        """Aligns the UTC normalized timeseries data to the input timezone.

        Since PyPSA can only handle UTC timeseries data, all timeseries data
        is normalized to UTC timezone during the construction of the PyPSA energy system.
        However, since the consumers expect to receive the results in the original timezone,
        the PyPSA results must be realigned with the input timezone.
        """
        if isinstance(pypsa_values, float):
            return pypsa_values
        return pypsa_utils.denormalize_dti_or_df(pypsa_values, dti.tz)

    @override
    def get_stored_energy(
        self, block: blocks.BaseBlock, dti: pd.DatetimeIndex
    ) -> optimization_problem.FloatOrTimeSeries:
        if not isinstance(block, blocks.ElectricFleetUnit) and not isinstance(block, blocks.StationaryBattery):
            raise ValueError(f"Cannot determine SoC for block {block.name} of type {type(block)}")

        pypsa_store_name = make_pypsa_label(block, "battery-store")

        normalized_dti = pypsa_utils.normalize_dti_or_df(dti)
        normalized_pypsa_dti = self._net.snapshots
        if len(normalized_dti) > len(normalized_pypsa_dti):
            # Extended DTI passed (oemof-style, n+1 points).
            # PyPSA's e is end-of-interval, so we reconstruct the n+1 series by
            # prepending e_initial as the first (start-of-horizon) value.
            pypsa_store_e = self._net.stores_t.e[pypsa_store_name]  # length n, end-of-interval

            e_initial = self._net.stores.at[pypsa_store_name, "e_initial"]  # scalar, start-of-horizon
            # e_initial is stored as a fraction of e_nom in PyPSA
            e_nom = self._net.stores.at[pypsa_store_name, "e_nom"]
            e_initial_mwh = e_initial * e_nom

            # Build the n+1 series: [e_initial, e[t0], e[t1], ..., e[tN-1]]
            extended_e = pd.concat(
                [
                    pd.Series([e_initial_mwh], index=[normalized_pypsa_dti[0]]),  # placeholder index, will be reindexed
                    pypsa_store_e,
                ]
            )
            extended_e.index = normalized_dti  # align to the extended oemof-style dti

            stored_energy = self._align_pypsa_values_to_dti(extended_e, dti)
        else:
            # Standard DTI: just return end-of-interval SoC aligned to snapshots
            pypsa_store_e = self._net.stores_t.e.loc[normalized_dti, pypsa_store_name]
            stored_energy = self._align_pypsa_values_to_dti(pypsa_store_e, dti)

        return stored_energy

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
    optimization_problem.Solver.GUROBI,
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

        if self._config.warmstart:
            self._model = self._net.optimize.create_model(include_objective_constant=False)
            self._warmstart_folder = tempfile.TemporaryDirectory()
        else:
            self._model = None
            self._warmstart_folder = None

    @property
    def pypsa_network(self) -> pypsa.Network:
        return self._net

    @override
    @classmethod
    def from_revoletion_scenario(
        cls,
        scenario: scn.Scenario,
        horizon: time.TimeFrame,
        logger: logging.Logger,
        config: optimization_problem.OptimizationProblemConfig | None = None,
    ) -> Self:
        if config is None:
            config = optimization_problem.OptimizationProblemConfig()

        visitor = PyPSABlockVisitor(
            horizon,
            logger,
            config.cost_eps,
            enable_investment=config.invest,
            enable_commitment=config.commitment,
        )
        pypsa_network = visitor.create_pypsa_network(scenario.block_registry)

        return cls(pypsa_network, logger, config)

    @override
    def solve(self) -> tuple[optimization_problem.OptimizationStatus, optimization_problem.OptimizationResult | None]:
        dti = self._net.snapshots

        status = self._do_solve(dti)
        if status != optimization_problem.OptimizationStatus.OPTIMAL:
            return status, None

        # PyPSA does not provide separate results like OMEOF.
        # Instead, the results are directly saved inside the network.
        return status, PypsaOptimizationResult(net=self._net)

    @override
    def solve_time_step(
        self, time_step: pd.DatetimeIndex
    ) -> tuple[optimization_problem.OptimizationStatus, optimization_problem.OptimizationResult | None]:
        normalized_time_step = pypsa_utils.normalize_dti_or_df(time_step)

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
        if status != optimization_problem.OptimizationStatus.OPTIMAL:
            return status, None

        # PyPSA does not provide separate results like OMEOF, and instead the results are directly saved inside the network.
        # Therefore, the network is just wrapped inside the `PypsaOptimizationResult`.
        return status, PypsaOptimizationResult(net=self._net)

    def _do_solve(self, dti: pd.DatetimeIndex) -> optimization_problem.OptimizationStatus:
        # Try to directly communicate the optimization problem to the HiGHS solver, without writing it to a file.
        # This should reduce the I/O interactions and significantly speed up optimizations for large scenarios.
        io_api = "direct" if self._config.solver == optimization_problem.Solver.HIGHS else None

        optimize_kwargs: dict[str, Any] = dict(
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

        if self._config.commitment:
            # When unit commitment is enabled, the optimization problem is transformed into a
            # Mixed-Integer problem which can be more costly to optimize.
            # We can improve performance by allowing 'less' optimal results.
            optimize_kwargs["mip_rel_gap"] = 0.01

        if self._warmstart_folder is not None:
            warmstart_file = f"{self._warmstart_folder.name}/basis.lp"
            optimize_kwargs["basis_fn"] = warmstart_file
            optimize_kwargs["warmstart_fn"] = warmstart_file

        if self._model:
            self._net._model = self._model
            with suppress_output():
                # Solve the created model.
                solver_status_str, termination_condition_str = self._net.optimize.solve_model(**optimize_kwargs)
            del self._net.model
        else:
            with suppress_output():
                solver_status_str, termination_condition_str = self._net.optimize(dti, **optimize_kwargs)

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
