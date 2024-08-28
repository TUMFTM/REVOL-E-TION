"""
custom_constraints.py

--- Description ---
This script defines additional constraints for the REVOL-E-TION toolset.

For further information, see readme

--- Created by ---
Brian Dietermann

--- File Information ---
coding:     utf-8
"""

import pyomo.environ as po

from src import blocks


class CustomConstraints:
    def __init__(self, scenario):
        self.scenario = scenario
        self.equal_invests = []

    def apply_constraints(self, model):

        model.CUSTOM_CONSTRAINTS = po.Block()
        # Apply additional constraints to equalize investment variables for bidirectional flows
        self.equate_invests(model)

        # Limit the sum of the power flows of different GridMarkets to the current power of the GridConnection
        self.limit_gridmarket_power(model)

        # Limit energy fed into grids and energy storages for which "res_only" is activated to renewable energies only
        self.renewables_only(model)

    def add_equal_invests(self, invests):
        # Add a list of investment variables represented as dicts containing the start and end node of a flow
        self.equal_invests.append(invests)

    def equate_invests(self, model):
        # Goal:     Several sizes (e.g. SystemCore's AC/DC and DC/AC converter, GridConnection sizes) can be forced to
        #           have the same value despite being optimized by independent Investment objects
        # Approach: Add a constraint to force a specified list of values to be equal
        model.CUSTOM_CONSTRAINTS.EQUATE_INVESTS = po.Block()

        def _equate_invest_variables(m, block, name, variables):
            def _equate_invest_variables_rule(block):
                expr = (variables[0] == variables[1])
                for var in variables[2:]:
                    expr = expr & (variables[0] == var)
                return expr

            setattr(block, name, po.Constraint(rule=_equate_invest_variables_rule))

        # Add additional user-specific constraints for investment variables
        for var_list in self.equal_invests:
            _equate_invest_variables(m=model,
                                     block=model.CUSTOM_CONSTRAINTS.EQUATE_INVESTS,
                                     name="_equal_".join([f'{var["in"]}_to_{var["out"]}' for var in var_list]),
                                     variables=[model.InvestmentFlowBlock.invest[var['in'], var['out'], 0]
                                                for var in var_list])

    def limit_gridmarket_power(self, model):
        # Goal:         Limit the sum of the power flows of different GridMarkets to the current power of the
        #               GridConnection. This ensures that all power being bought or sold has to reach the local energy
        #               system and avoids unlimited trading with energy on the different markets without any power
        #               limitations. As this model focuses on modeling a local energy system, trading without any
        #               physical power flow is not considered.
        # Approach:     1.  For each direction (buy/sell = g2mg/mg2g) sum up all power flows of different GridMarkets
        #                   connected to the same GridConnection.
        #               2.  Constrain the sum of GridMarkets' power flows in each direction to not exceed the current
        #                   corresponding power flow of the GridConnection considering the parallel flows connecting the
        #                   grid bus to the main bus to allow peakshaving.

        model.CUSTOM_CONSTRAINTS.LIMIT_GRIDMARKET_POWER = po.Block()

        def _limit_flows(m, block, name, flows_markets, flows_grid):
            def _limit_flows_rule(block):
                for p, ts in m.TIMEINDEX:
                    pwr_market = sum(m.flow[fi, fo, p, ts] for fi, fo in flows_markets)
                    pwr_grid = sum(m.flow[fi, fo, p, ts] for fi, fo in flows_grid)
                    expr = pwr_market == pwr_grid

                    if expr is not True:
                        getattr(block, name).add((p, ts), expr)

            setattr(block, name, po.Constraint(m.TIMEINDEX, noruleinit=True))
            setattr(block, name + "_build", po.BuildAction(rule=_limit_flows_rule))

        # Apply constraints for every GridConnection
        for grid in [block for block in self.scenario.blocks.values() if isinstance(block, blocks.GridConnection)]:
            _limit_flows(m=model,
                         block=model.CUSTOM_CONSTRAINTS.LIMIT_GRIDMARKET_POWER,
                         name=f'limit_{grid.name}_g2mg_markets',
                         flows_markets=[(market.src, grid.bus) for market in grid.markets.values()],
                         flows_grid=[(grid.bus, converter) for converter in grid.outflow.values()])

            _limit_flows(m=model,
                         block=model.CUSTOM_CONSTRAINTS.LIMIT_GRIDMARKET_POWER,
                         name=f'limit_{grid.name}_mg2g_markets',
                         flows_markets=[(grid.bus, market.snk) for market in grid.markets.values()],
                         flows_grid=[(converter, grid.bus) for converter in grid.inflow.values()])

    def renewables_only(self, model):
        # Goal:         For all specified blocks restrict feed_in of energy into the block to renewable energy only
        # Definition:   Renewable energy is energy generated by PV and wind sources and energy generated by those
        #               blocks which is stored in a storage which only allows renewable energy to be stored
        # Approach:     1.  Sum up all power originating renewable energy blocks flowing into both SystemCore buses
        #               2.  Split those sums into power remaining on the bus and power converted to the other bus
        #               3.  Limit the converted renewable power to the power converted at the converter
        #               4.  Limit the sum of renewable power fed into all restricted components connected to each bus
        #                   to the sum of renewable power present on each bus (renewable power directly fed into the bus
        #                   and renewable power converted to the bus multiplied by the SystemCore converter efficiency)

        # Add new block within the CUSTOM_CONSTRAINTS block to store all constraints related to renewable energy only
        model.CUSTOM_CONSTRAINTS.RES_ONLY = po.Block()
        # Add the variables to store the renewable power flows (format: res_[from bus][to bus])
        for var_name in ['res_acac', 'res_acdc', 'res_dcac', 'res_dcdc']:
            setattr(model.CUSTOM_CONSTRAINTS.RES_ONLY,
                    var_name,
                    po.Var(model.TIMEINDEX, within=po.NonNegativeReals))

        # Get discharging flows of all StationaryEnergyStorages which only allow storing renewable energy
        # (DC connection is hardcoded -> has to be modified if StationaryEnergySystem.bus_connected is changed)
        storage_flows = [(block.ess, block.bus_connected) for block in self.scenario.blocks.values() if
                         isinstance(block, blocks.StationaryEnergyStorage) and block.res_only]

        # Get flows of all components connected to each SystemCore bus which only allow feed-in of renewable energy
        flows_res_from_bus = {
            'ac': [(market.parent.bus, market.snk) for block in self.scenario.blocks.values() if
                   isinstance(block, blocks.GridConnection) for market in block.markets.values() if
                   market.res_only],
            'dc': [(fo, fi) for fi, fo in storage_flows]  # invert discharging flows to charging flows
        }

        # Get all renewable power flows
        flows_res_to_bus = {
            'ac': [(block.outflow, block.bus_connected) for block in self.scenario.blocks.values() if
                   isinstance(block, blocks.RenewableInvestBlock) and 'ac' in block.bus_connected.label],
            'dc': [(block.outflow, block.bus_connected) for block in self.scenario.blocks.values() if
                   isinstance(block, blocks.RenewableInvestBlock) and 'dc' in block.bus_connected.label] + storage_flows
        }

        def _sum_res(m, block, name, sum_flow, split_flows):
            def _sum_res_rule(block):  # not sure why m is not passed but block, but now it works
                for p, ts in m.TIMEINDEX:
                    res_sum = sum(m.flow[fi, fo, p, ts] for fi, fo in sum_flow)
                    res_summands = sum(var[p, ts] for var in split_flows)
                    expr = res_sum == res_summands

                    if expr is not True:
                        getattr(block, name).add((p, ts), expr)

            setattr(block, name, po.Constraint(model.TIMEINDEX, noruleinit=True))
            setattr(block, name + "_build", po.BuildAction(rule=_sum_res_rule))

        # define res_ac as sum of res_acac and res_acdc
        _sum_res(m=model,
                 block=model.CUSTOM_CONSTRAINTS.RES_ONLY,
                 name='sum_res_ac',
                 sum_flow=flows_res_to_bus['ac'],
                 split_flows=[model.CUSTOM_CONSTRAINTS.RES_ONLY.res_acac, model.CUSTOM_CONSTRAINTS.RES_ONLY.res_acdc])
        # define res_dc as sum of res_dcac and res_dcdc
        _sum_res(m=model,
                 block=model.CUSTOM_CONSTRAINTS.RES_ONLY,
                 name='sum_res_dc',
                 sum_flow=flows_res_to_bus['dc'],
                 split_flows=[model.CUSTOM_CONSTRAINTS.RES_ONLY.res_dcac, model.CUSTOM_CONSTRAINTS.RES_ONLY.res_dcdc])

        def _limit_res_to_conv(m, block, name, conv_flow, res_flow):
            def _limit_res2conv_rule(block):
                for p, ts in m.TIMEINDEX:
                    expr = m.flow[conv_flow[0], conv_flow[1], p, ts] >= res_flow[p, ts]

                    if expr is not True:
                        getattr(block, name).add((p, ts), expr)

            setattr(block, name, po.Constraint(model.TIMEINDEX, noruleinit=True))
            setattr(block, name + "_build", po.BuildAction(rule=_limit_res2conv_rule))

        # limit flow of renewable power from AC to DC to the maximum power of the AC/DC converter in SystemCore
        _limit_res_to_conv(m=model,
                           block=model.CUSTOM_CONSTRAINTS.RES_ONLY,
                           name='limit_res_acdc_to_conv',
                           conv_flow=(self.scenario.blocks['core'].ac_bus, self.scenario.blocks['core'].ac_dc),
                           res_flow=model.CUSTOM_CONSTRAINTS.RES_ONLY.res_acdc)
        # limit flow of renewable power from DC to AC to the maximum power of the DC/AC converter in SystemCore
        _limit_res_to_conv(m=model,
                           block=model.CUSTOM_CONSTRAINTS.RES_ONLY,
                           name='limit_res_dcac_to_conv',
                           conv_flow=(self.scenario.blocks['core'].dc_bus, self.scenario.blocks['core'].dc_ac),
                           res_flow=model.CUSTOM_CONSTRAINTS.RES_ONLY.res_dcac)

        def _limit_feed_in(m, block, name, flows_feed_in, flows_res, eff_conv):
            def _limit_feed_in_rule(block):
                for p, ts in m.TIMEINDEX:
                    power_res_feed_in = sum(m.flow[fi, fo, p, ts] for fi, fo in flows_feed_in)
                    power_res_available = sum(flow_res[p, ts] * eff for flow_res, eff in zip(flows_res, eff_conv))
                    expr = power_res_feed_in <= power_res_available

                    if expr is not True:
                        getattr(block, name).add((p, ts), expr)

            setattr(block, name, po.Constraint(model.TIMEINDEX, noruleinit=True))
            setattr(block, name + "_build", po.BuildAction(rule=_limit_feed_in_rule))

        # limit feed-in of renewable power from the AC bus to components connected to the AC-bus considering the
        # SystemCore's converter efficiency
        _limit_feed_in(m=model,
                       block=model.CUSTOM_CONSTRAINTS.RES_ONLY,
                       name='limit_res_ac_feed_in',
                       flows_feed_in=flows_res_from_bus['ac'],
                       flows_res=[model.CUSTOM_CONSTRAINTS.RES_ONLY.res_acac,
                                  model.CUSTOM_CONSTRAINTS.RES_ONLY.res_dcac],
                       eff_conv=[1, self.scenario.blocks['core'].eff_dcac])

        # limit feed-in of renewable power from the DC bus to components connected to the DC-bus considering the
        # SystemCore's converter efficiency
        _limit_feed_in(m=model,
                       block=model.CUSTOM_CONSTRAINTS.RES_ONLY,
                       name='limit_res_dc_feed_in',
                       flows_feed_in=flows_res_from_bus['dc'],
                       flows_res=[model.CUSTOM_CONSTRAINTS.RES_ONLY.res_dcac,
                                  model.CUSTOM_CONSTRAINTS.RES_ONLY.res_dcdc],
                       eff_conv=[self.scenario.blocks['core'].eff_acdc, 1])
