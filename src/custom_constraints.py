"""
custom_constraints.py

--- Description ---
This script defines additional constraints for the REVOL-E-TION toolset.

For further information, see readme

--- Created by ---
Brian Dietermann

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

import pyomo.environ as po

from src import blocks


class CustomConstraints:
    def __init__(self, scenario):
        self.scenario = scenario
        self.equal_invests = []

    def apply_constraints(self, model):
        # Apply additional constraints to equalize investment variables for bidirectional flows
        self.equate_invests(model)

        # Limit the sum of the power flows of different GridMarkets to the current power of the GridConnection
        self.limit_gridmarket_power(model)

        # Limit energy feed into the grid (and to energy storage) to renewable energies only
        self.renewables_only(model, self.scenario)

    def add_equal_invests(self, invests):
        # Add a list of investment variables represented as dicts containing the start and end node of a flow
        self.equal_invests.append(invests)

    def equate_invests(self, model):
        # Add additional user-specific constraints for investment variables
        for var_list in self.equal_invests:
            self.equate_mult_vars(model=model,
                                  variables=[model.InvestmentFlowBlock.invest[var['in'], var['out'], 0]
                                             for var in var_list])

    @staticmethod
    def equate_mult_vars(model, variables, name=None):
        r"""
        Adds a constraint to the given model that sets multiple variables to equal.

        Parameters
        ----------
        variables : list of pyomo.environ.Var
            List of variables which all should be set equal
        name : str
            Optional name for the equation e.g. in the LP file. By default the
            name is: equate + string representation of variables.
        model : oemof.solph._models.Model
            Model to which the constraint is added.
        """  # noqa: E501

        if name is None:
            name = "_".join(["equate"] + [str(var) for var in variables])

        def equate_variables_rule(m):
            return [variables[0] == var for var in variables[1:]]

        setattr(model, name, po.ConstraintList(rule=equate_variables_rule))

    def limit_gridmarket_power(self, model):
        # Add additional constraints to limit the sum of the power flows of different GridMarkets to the current power
        # of the GridConnection
        for grid_connection in [block for block in self.scenario.blocks.values()
                                if isinstance(block, blocks.GridConnection)]:
            connection_g2mg = [(grid_connection.bus, converter) for converter in grid_connection.outflow.values()]
            connection_mg2g = [(converter, grid_connection.bus) for converter in grid_connection.inflow.values()]

            markets_g2mg = [(market.src, grid_connection.bus) for market in grid_connection.markets.values()]
            markets_mg2g = [(grid_connection.bus, market.snk) for market in grid_connection.markets.values()]

            self.limit_flows(model=model, flows=markets_g2mg, upper_bound=connection_g2mg, name='limit_g2mg_markets')

            self.limit_flows(model=model, flows=markets_mg2g, upper_bound=connection_mg2g, name='limit_mg2g_markets')

    @staticmethod
    def limit_flows(model, flows, upper_bound, name=None):
        r"""
            Adds a constraint to the given model that limits the sum of flows to upper bound.
            Upper bound can be given as a scalar or a list of flows.
            If a list is given all flows in the list are summed up to get the limit.
    
            Parameters
            ----------
            model : oemof.solph._models.Model
                Model to which the constraint is added.
            flows : list of Flows 
                List of flows which sum is limited.
            upper_bound: list of Flows or Scalar
                List of flows which are summed up or a scalar. The sum or the scalar is then used to limit the sum of
                the flows given in the argument flow
            name : str
                Optional name for the equation e.g. in the LP file.
            """  # noqa: E501
        def _limit_flows(m):
            for p, ts in m.TIMEINDEX:
                flows_sum = sum(m.flow[fi, fo, p, ts] for fi, fo in flows)
                bound = sum(m.flow[fi, fo, p, ts] for fi, fo in upper_bound) if isinstance(upper_bound,
                                                                                           list) else upper_bound
                expr = flows_sum <= bound

                if expr is not True:
                    getattr(m, name).add((p, ts), expr)

        if name is None:
            name = '_'.join(
                ['limit_sum_of'] + [str(flow) for flow in flows] +
                ['to_upper_bound'] +
                (['sum_of'] + [str(flow) for flow in upper_bound]
                 if isinstance(upper_bound, list) else [str(upper_bound)])
            )

        setattr(model, name, po.Constraint(model.TIMEINDEX, noruleinit=True))
        setattr(model, name + "_build", po.BuildAction(rule=_limit_flows))

    def renewables_only(self, model, scenario):
        # Apply additional constraints to limit energy feed into the grid to renewable energy generation
        grid_flows = [(block.bus_connected, block.snk) for block in scenario.blocks.values() if
                      isinstance(block, blocks.GridConnection) and getattr(block, 'res_only', False)]
        pv_flows = [(block.outflow, block.bus_connected) for block in scenario.blocks.values() if
                    isinstance(block, (blocks.PVSource))]
        wind_flows = [(block.outflow, block.bus_connected) for block in scenario.blocks.values() if
                      isinstance(block, (blocks.WindSource))]
        ess_in_flows = [(block.bus_connected, block.ess) for block in scenario.blocks.values() if
                        isinstance(block, (blocks.StationaryEnergyStorage)) and getattr(block, 'pv_only', False)]
        ess_out_flows = [(block.ess, block.bus_connected) for block in scenario.blocks.values() if
                         isinstance(block, (blocks.StationaryEnergyStorage)) and getattr(block, 'pv_only', False)]
        sys_core = [block for block in scenario.blocks.values() if isinstance(block, blocks.SystemCore)][0]
        if grid_flows:
            if ess_in_flows:
                # ensure that no commodity system is connected to the DC bus
                if any([getattr(block, 'bus_connected', False) == sys_core.dc_bus for block in scenario.blocks.values()
                        if
                        isinstance(block, blocks.CommoditySystem)]):
                    # ToDo: only exit scenario, not the whole execution
                    print('Error: Commodity systems are connected to the DC bus. This is not allowed if "pv_only" is '
                          'activated for a stationary storage. Exiting.')
                    quit()
                # force AC/DC to be 0 for all timesteps
                self.set_flows_to_zero(model=model,
                                       flows=[(sys_core.ac_bus, sys_core.ac_dc)],
                                       name='set_acdc_to_zero')
            self.res_only(model=model,
                          grid_flows=grid_flows,
                          pv_flows=pv_flows,
                          wind_flows=wind_flows,
                          ess_in_flows=ess_in_flows,
                          ess_out_flows=ess_out_flows,
                          syscore_dcac_eff=sys_core.dcac_eff,
                          name='res_only')

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
        """ # noqa: E501

        def _set_to_zero_rule(m):
            for p, ts in m.TIMEINDEX:
                for fi, fo in flows:
                    expr = m.flow[fi, fo, p, ts] == 0
                    if expr is not True:
                        getattr(m, name).add((p, ts), expr)

        setattr(model, name, po.Constraint(model.TIMEINDEX, noruleinit=True))
        setattr(model, name + "_build", po.BuildAction(rule=_set_to_zero_rule))

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
        """ # noqa: E501

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

        setattr(model, name, po.Constraint(model.TIMEINDEX, noruleinit=True))
        setattr(model, name + "_build", po.BuildAction(rule=_limit_grid_to_res_only_rule))

        return model