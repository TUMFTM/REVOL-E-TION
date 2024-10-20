#!/usr/bin/env python3

import pyomo.environ as po
from pyomo import environ as po2

from revoletion import blocks


class CustomConstraints:
    def __init__(self, scenario):
        self.scenario = scenario
        self.equal_invests = []
        self.invest_costs = {'flow': [], 'storage': []}

    def apply_constraints(self, model):
        # Add pyomo block to model to store custom constraints
        model.CUSTOM_CONSTRAINTS = po.Block()
        # Apply additional constraints to equalize investment variables for bidirectional flows
        self.equate_invests(model)

        # Limit the sum of the power flows of different GridMarkets to the current power of the GridConnection
        self.limit_pwr_gridmarket(model)

        # Limit energy fed into grids and energy storages for which "res_only" is activated to renewable energies only
        self.renewables_only(model)

        # Force all charged energy into the commodity's storage
        self.external_charging_to_storage(model)

        # Limit initial investment costs
        self.limit_invest_costs(model)

    def add_equal_invests(self, invests):
        # Add a list of investment variables represented as dicts containing the start and end node of a flow
        self.equal_invests.append(invests)

    def add_invest_costs(self, invest, capex_spec, invest_type):
        # needs to be a custom solution, as peakshaving also uses investement objects but should not be considered
        if invest_type == 'flow':
            self.invest_costs[invest_type].append({'fi': invest[0], 'fo': invest[1], 'capex_spec': capex_spec})
        elif invest_type == 'storage':
            self.invest_costs[invest_type].append({'so': invest[0], 'capex_spec': capex_spec})

    def equate_invests(self, model):
        # Goal:     Several sizes (e.g. SystemCore's AC/DC and DC/AC converter, GridConnection sizes) can be forced to
        #           have the same value despite being optimized by independent Investment objects
        # Approach: Add a constraint to force a specified list of values to be equal
        model.CUSTOM_CONSTRAINTS.EQUATE_INVESTS = po.Block()

        def _equate_invest_variables(m, block, name, variables):
            def _equate_invest_variables_rule(block):
                return variables[0] == variables[1]

            setattr(block, name, po.Constraint(rule=_equate_invest_variables_rule))

        # Add additional user-specific constraints for investment variables
        for var_list in self.equal_invests:
            for var_equal in var_list[1:]:
                _equate_invest_variables(
                    m=model,
                    block=model.CUSTOM_CONSTRAINTS.EQUATE_INVESTS,
                    name="_equal_".join([f'{var["in"]}_to_{var["out"]}' for var in [var_list[0], var_equal]]),
                    variables=[model.InvestmentFlowBlock.invest[var['in'], var['out'], 0]
                               for var in [var_list[0], var_equal]])

    def limit_pwr_gridmarket(self, model):
        # Goal:         Limit the sum of the power flows of different GridMarkets to the current power of the
        #               GridConnection. This ensures that all power being bought or sold has to reach the local energy
        #               system and avoids unlimited trading with energy on the different markets without any power
        #               limitations. As this model focuses on modeling a local energy system, trading without any
        #               physical power flow is not considered.
        # Approach:     1.  For each direction (buy/sell = g2s/s2g) sum up all power flows of different GridMarkets
        #                   connected to the same GridConnection.
        #               2.  Constrain the sum of GridMarkets' power flows in each direction to not exceed the current
        #                   corresponding power flow of the GridConnection considering the parallel flows connecting the
        #                   grid bus to the main bus to allow peakshaving.

        model.CUSTOM_CONSTRAINTS.LIMIT_PWR_GRIDMARKET = po.Block()

        def _limit_flows(m, block, name, flows_markets, flows_grid):
            def _limit_flows_rule(block):
                for p, ts in m.TIMEINDEX:
                    pwr_market = sum(m.flow[fi, fo, ts] for fi, fo in flows_markets)
                    pwr_grid = sum(m.flow[fi, fo, ts] for fi, fo in flows_grid)
                    expr = pwr_market == pwr_grid

                    if expr is not True:
                        getattr(block, name).add((p, ts), expr)

            setattr(block, name, po.Constraint(m.TIMEINDEX, noruleinit=True))
            setattr(block, name + "_build", po.BuildAction(rule=_limit_flows_rule))

        # Apply constraints for every GridConnection
        for grid in [block for block in self.scenario.blocks.values() if isinstance(block, blocks.GridConnection)]:
            _limit_flows(m=model,
                         block=model.CUSTOM_CONSTRAINTS.LIMIT_PWR_GRIDMARKET,
                         name=f'limit_{grid.name}_g2s_markets',
                         flows_markets=[(market.src, grid.bus) for market in grid.markets.values()],
                         flows_grid=[(grid.bus, converter) for converter in grid.outflow.values()])

            _limit_flows(m=model,
                         block=model.CUSTOM_CONSTRAINTS.LIMIT_PWR_GRIDMARKET,
                         name=f'limit_{grid.name}_s2g_markets',
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
        model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY = po.Block()
        # Add the variables to store the renewable power flows (format: res_[from bus][to bus])
        for var_name in ['pwr_res_acac', 'pwr_res_acdc', 'pwr_res_dcac', 'pwr_res_dcdc']:
            setattr(model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
                    var_name,
                    po.Var(model.TIMEINDEX, within=po.NonNegativeReals))

        # Get discharging flows of all StationaryEnergyStorages which only allow storing renewable energy
        storage_flows_ac = [(block.bus, block.bus_connected) for block in self.scenario.blocks.values() if
                            isinstance(block, blocks.StationaryEnergyStorage) and block.res_only and block.system == 'ac']
        storage_flows_dc = [(block.bus, block.bus_connected) for block in self.scenario.blocks.values() if
                            isinstance(block, blocks.StationaryEnergyStorage) and block.res_only and block.system == 'dc']

        # Get flows of all components connected to each SystemCore bus which only allow feed-in of renewable energy
        flows_res_from_bus = {
            'ac': [(market.parent.bus, market.snk) for block in self.scenario.blocks.values() if
                   isinstance(block, blocks.GridConnection) for market in block.markets.values() if
                   market.res_only] + [(fo, fi) for fi, fo in storage_flows_ac],
            'dc': [(fo, fi) for fi, fo in storage_flows_dc]  # invert discharging flows to charging flows
        }

        # Get all renewable power flows
        flows_res_to_bus = {
            'ac': [(block.outflow, block.bus_connected) for block in self.scenario.blocks.values() if
                   isinstance(block, blocks.RenewableInvestBlock) and 'ac' in block.bus_connected.label] + storage_flows_ac,
            'dc': [(block.outflow, block.bus_connected) for block in self.scenario.blocks.values() if
                   isinstance(block, blocks.RenewableInvestBlock) and 'dc' in block.bus_connected.label] + storage_flows_dc
        }

        def _sum_res(m, block, name, sum_flow, split_flows):
            def _sum_res_rule(block):  # not sure why m is not passed but block, but now it works
                for p, ts in m.TIMEINDEX:
                    res_sum = sum(m.flow[fi, fo, ts] for fi, fo in sum_flow)
                    res_summands = sum(var[p, ts] for var in split_flows)
                    expr = res_sum == res_summands

                    if expr is not True:
                        getattr(block, name).add((p, ts), expr)

            setattr(block, name, po.Constraint(model.TIMEINDEX, noruleinit=True))
            setattr(block, name + "_build", po.BuildAction(rule=_sum_res_rule))

        # define res_ac as sum of pwr_res_acac and pwr_res_acdc
        _sum_res(m=model,
                 block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
                 name='sum_res_ac',
                 sum_flow=flows_res_to_bus['ac'],
                 split_flows=[model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_acac,
                              model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_acdc])
        # define res_dc as sum of pwr_res_dcac and pwr_res_dcdc
        _sum_res(m=model,
                 block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
                 name='sum_res_dc',
                 sum_flow=flows_res_to_bus['dc'],
                 split_flows=[model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcac,
                              model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcdc])

        def _limit_res_to_conv(m, block, name, conv_flow, res_flow):
            def _limit_res2conv_rule(block):
                for p, ts in m.TIMEINDEX:
                    expr = m.flow[conv_flow[0], conv_flow[1], ts] >= res_flow[p, ts]

                    if expr is not True:
                        getattr(block, name).add((p, ts), expr)

            setattr(block, name, po.Constraint(model.TIMEINDEX, noruleinit=True))
            setattr(block, name + "_build", po.BuildAction(rule=_limit_res2conv_rule))

        # limit flow of renewable power from AC to DC to the maximum power of the AC/DC converter in SystemCore
        _limit_res_to_conv(m=model,
                           block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
                           name='limit_pwr_res_acdc_to_conv',
                           conv_flow=(self.scenario.blocks['core'].ac_bus, self.scenario.blocks['core'].ac_dc),
                           res_flow=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_acdc)
        # limit flow of renewable power from DC to AC to the maximum power of the DC/AC converter in SystemCore
        _limit_res_to_conv(m=model,
                           block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
                           name='limit_pwr_res_dcac_to_conv',
                           conv_flow=(self.scenario.blocks['core'].dc_bus, self.scenario.blocks['core'].dc_ac),
                           res_flow=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcac)

        def _limit_feed_in(m, block, name, flows_feed_in, flows_res, eff_conv):
            def _limit_feed_in_rule(block):
                for p, ts in m.TIMEINDEX:
                    pwr_res_feed_in = sum(m.flow[fi, fo, ts] for fi, fo in flows_feed_in)
                    pwr_res_available = sum(flow_res[p, ts] * eff for flow_res, eff in zip(flows_res, eff_conv))
                    expr = pwr_res_feed_in <= pwr_res_available

                    if expr is not True:
                        getattr(block, name).add((p, ts), expr)

            setattr(block, name, po.Constraint(model.TIMEINDEX, noruleinit=True))
            setattr(block, name + "_build", po.BuildAction(rule=_limit_feed_in_rule))

        # limit feed-in of renewable power from the AC bus to components connected to the AC-bus considering the
        # SystemCore's converter efficiency
        _limit_feed_in(m=model,
                       block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
                       name='limit_res_ac_feed_in',
                       flows_feed_in=flows_res_from_bus['ac'],
                       flows_res=[model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_acac,
                                  model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcac],
                       eff_conv=[1, self.scenario.blocks['core'].eff_dcac])

        # limit feed-in of renewable power from the DC bus to components connected to the DC-bus considering the
        # SystemCore's converter efficiency
        _limit_feed_in(m=model,
                       block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
                       name='limit_res_dc_feed_in',
                       flows_feed_in=flows_res_from_bus['dc'],
                       flows_res=[model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcac,
                                  model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcdc],
                       eff_conv=[self.scenario.blocks['core'].eff_acdc, 1])

    def external_charging_to_storage(self, model):
        # Goal:         Force all external charged power to flow into the commodity's storage.
        #               This is necessary to ensure that the external charged power is not consumed without .
        # Approach:     For each commodity ensure that the sum of all three charging powers is equal to the storage inflow

        model.CUSTOM_CONSTRAINTS.EXTERNAL_CHARGING_STORAGE = po.Block()

        def _equal_flows(m, block, name, flows_charging, flows_storage):
            def _equal_flows_rule(block):
                for p, ts in m.TIMEINDEX:
                    pwr_charging = sum(m.flow[fi, fo, ts] for fi, fo in flows_charging)
                    pwr_storage = sum(m.flow[fi, fo, ts] for fi, fo in flows_storage)
                    expr = pwr_charging == pwr_storage

                    if expr is not True:
                        getattr(block, name).add((p, ts), expr)

            setattr(block, name, po.Constraint(m.TIMEINDEX, noruleinit=True))
            setattr(block, name + "_build", po.BuildAction(rule=_equal_flows_rule))

        # Apply constraints for every MobileCommodity
        for cs in [block for block in self.scenario.blocks.values() if isinstance(block, blocks.CommoditySystem)]:
            for commodity in cs.commodities.values():
                _equal_flows(m=model,
                             block=model.CUSTOM_CONSTRAINTS.EXTERNAL_CHARGING_STORAGE,
                             name=f'limit_{commodity.name}_external_charging_to_storage',
                             flows_charging=[(commodity.inflow, commodity.bus),
                                            (commodity.conv_ext_ac, commodity.bus),
                                            (commodity.conv_ext_dc, commodity.bus)],
                             flows_storage=[(commodity.bus, commodity.ess)])

    def limit_invest_costs(self, model):
        # Goal:     Limit all initial investment costs to a specified value (neglect peakshaving investments)
        # Approach: Add a constraint adding all initial investment costs and limiting the sum to the specified value
        model.CUSTOM_CONSTRAINTS.LIMIT_INVESTS = po.Block()

        def _limit_invests(m, block, name):
            def _limit_invest_rule(block):
                expr = 0

                # Add investment costs for all flow objects
                expr += sum(m.InvestmentFlowBlock.invest[invest_flow['fi'], invest_flow['fo'], 0] *
                            invest_flow['capex_spec'] for invest_flow in self.invest_costs['flow'])

                # Add investment costs for all storage objects
                expr += sum(m.GenericInvestmentStorageBlock.invest[invest_storage['so'], 0] *
                            invest_storage['capex_spec'] for invest_storage in self.invest_costs['storage'])

                return expr <= self.scenario.invest_max

            setattr(block, name, po.Constraint(rule=_limit_invest_rule))

        # Add additional user-specific constraints for investment cost limit
        if self.scenario.invest_max is not None:
            _limit_invests(m=model,
                          block=model.CUSTOM_CONSTRAINTS.LIMIT_INVESTS,
                          name='limit_invest_costs')
