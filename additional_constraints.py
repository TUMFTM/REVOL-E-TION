import oemof.solph as solph
import pyomo.environ as po
import blocks

###############################################################################
# Constraints definitions
###############################################################################


class Constraints:
    @staticmethod
    def set_flows_to_zero(model, flows, name='set_to_zero'):
        r"""
        Adds a constraint to the given model that ensures that the sum of a group of
        flows multiplied with a factor is equal or less than the sum of another group of flows for each timestep.
        Modified from the original equate_flows() function.

        Parameters
        ----------
        model : oemof.solph.Model
            Model to which the constraint is added.
        flows : iterable
            Group of flows that should be set to zero for every timestep.
        name : str, default='res_only'
            Name for the equation e.g. in the LP file.

        Returns
        -------
        the updated model.
        """

        def _set_to_zero_rule(m):
            for p, ts in m.TIMEINDEX:
                for fi, fo in flows:
                    expr = m.flow[fi, fo, p, ts] == 0
                    if expr is not True:
                        getattr(m, name).add((p, ts), expr)

        setattr(
            model,
            name,
            po.Constraint(model.TIMEINDEX, noruleinit=True),
        )
        setattr(
            model,
            name + "_build",
            po.BuildAction(rule=_set_to_zero_rule),
        )

        return model

    @staticmethod
    def res_only(model, grid_flows, pv_flows, wind_flows, ess_in_flows, ess_out_flows, syscore_dcac_eff,
                 name='res_only'):
        r"""
        Adds a constraint to the given model that ensures that the sum of a group of
        flows multiplied with a factor is equal or less than the sum of another group of flows for each timestep.
        Modified from the original equate_flows() function.

        Parameters
        ----------
        model : oemof.solph.Model
            Model to which the constraint is added.
        grid_flows : iterable
            Group of flows that feed energy into the grid.
        pv_flows : iterable
            Group of DC-connected pv flows.
        wind_flows : iterable
            Group of AC-connected wind flows.
        ess_in_flows : iterable
            Group of flows from DC-connected stationary energy storage systems into the local mini-grid.
        ess_out_flows : iterable
            Group of flows from the local mini-grid into DC-connected stationary energy storage systems.
        syscore_dcac_eff : float
            Efficiency of the DC/AC conversion in the system core.
        name : str, default='res_only'
            Name for the equation e.g. in the LP file.

        Returns
        -------
        the updated model.
        """

        def _limit_grid_to_res_only_rule(m):
            for p, ts in m.TIMEINDEX:
                sum_grid = sum(m.flow[fi, fo, p, ts] for fi, fo in grid_flows)
                sum_pv = sum(m.flow[fi, fo, p, ts] for fi, fo in pv_flows)
                sum_wind = sum(m.flow[fi, fo, p, ts] for fi, fo in wind_flows)
                sum_ess_in = sum(m.flow[fi, fo, p, ts] for fi, fo in ess_in_flows)
                sum_ess_out = sum(m.flow[fi, fo, p, ts] for fi, fo in ess_out_flows)
                expr = sum_grid <= syscore_dcac_eff * (sum_pv + sum_ess_in - sum_ess_out) + sum_wind
                if expr is not True:
                    getattr(m, name).add((p, ts), expr)

        setattr(
            model,
            name,
            po.Constraint(model.TIMEINDEX, noruleinit=True),
        )
        setattr(
            model,
            name + "_build",
            po.BuildAction(rule=_limit_grid_to_res_only_rule),
        )

        return model


def renewables_only(model, scenario):
    # Apply additional constraints to limit energy feed into the grid to renewable energy generation
    grid_flows = [(block.connected_bus, block.snk) for block in scenario.blocks.values() if
                  isinstance(block, blocks.GridConnection) and getattr(block, 'res_only', False)]
    pv_flows = [(block.outflow, block.connected_bus) for block in scenario.blocks.values() if
                isinstance(block, (blocks.PVSource))]
    wind_flows = [(block.outflow, block.connected_bus) for block in scenario.blocks.values() if
                isinstance(block, (blocks.WindSource))]
    ess_in_flows = [(block.connected_bus, block.ess) for block in scenario.blocks.values() if
                    isinstance(block, (blocks.StationaryEnergyStorage)) and getattr(block, 'pv_only', False)]
    ess_out_flows = [(block.ess, block.connected_bus) for block in scenario.blocks.values() if
                     isinstance(block, (blocks.StationaryEnergyStorage)) and getattr(block, 'pv_only', False)]
    sys_core = [block for block in scenario.blocks.values() if isinstance(block, blocks.SystemCore)][0]
    if grid_flows:
        if ess_in_flows:
            # ensure that no commodity system is connected to the DC bus
            if any([getattr(block, 'connected_bus', False) == sys_core.dc_bus for block in scenario.blocks.values() if
                    isinstance(block, blocks.CommoditySystem)]):
                # ToDo: only exit scenario, not the whole execution
                print('Error: Commodity systems are connected to the DC bus. This is not allowed if "pv_only" is '
                      'activated for a stationary storage. Exiting.')
                quit()
            # force AC/DC to be 0 for all timesteps
            Constraints.set_flows_to_zero(model=model,
                                          flows=[(sys_core.ac_bus, sys_core.ac_dc)],
                                          name='set_acdc_to_zero')
        Constraints.res_only(model=model,
                             grid_flows=grid_flows,
                             pv_flows=pv_flows,
                             wind_flows=wind_flows,
                             ess_in_flows=ess_in_flows,
                             ess_out_flows=ess_out_flows,
                             syscore_dcac_eff=sys_core.dcac_eff,
                             name='res_only')


def equal_sizing(model, scenario):
    # Add additional user-specific constraints for investment variables
    for equal_variable in scenario.equal_variables:
        # var1 * factor = var2
        solph.constraints.equate_variables(model=model,
                                           var1=model.InvestmentFlowBlock.invest[
                                               equal_variable['var1']['in'], equal_variable['var1']['out'], 0],
                                           var2=model.InvestmentFlowBlock.invest[
                                               equal_variable['var2']['in'], equal_variable['var2']['out'], 0],
                                           factor1=equal_variable['factor'])


def apply_additional_constraints(model, scenario):
    # Apply additional constraints to equalize investment variables for bidirectional flows
    equal_sizing(model, scenario)

    # Apply additional constraints to limit energy feed into the grid (and to energy storage) to renewable energies only
    renewables_only(model, scenario)
