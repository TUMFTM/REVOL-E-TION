#!/usr/bin/env python3

from dataclasses import dataclass, field

import pyomo.environ as po
from oemof import solph


@dataclass
class EquateFlowParams:
    """
    Dataclass storing the arguments which are later passed to oemof.solph.constraints.equate_flows()
    This ensures
          sum(flows1) * factor1 = sum(flows2)
    for all timesteps
    """

    flows1: list = field(default_factory=list)
    flows2: list = field(default_factory=list)
    factor1: float = 1.0
    name: str | None = None

    @property
    def dict(self) -> dict:
        if not self.flows1 or not self.flows2:
            raise ValueError(f"None of flows1 and flows2 is allowed to be empty! Pass flows to {self.name}")
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class EquateInvestParams:
    """
    Dataclass storing the arguments which are later passed to oemof.solph.constraints.equate_variables()
    This ensures
          invests[0] = invests[1] = ... = invests[n]
    """

    invests: list = field(default_factory=list)
    name: str | None = None

    @property
    def dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class _BaseParamsDict(dict):
    """
    Dictionary, which auto-creates missing entries and uses the key to initialize the entries value (therefore no defaultdict is used).
    """

    _FACTORY = None  # to be defined by subclasses

    @staticmethod
    def _validate_key(key):
        """
        ensure that all keys are a tuple of two strings
        """
        if not (isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, str) for k in key)):
            raise TypeError("Key must be a tuple of two strings, e.g. ('node1', 'node2')")

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def __missing__(self, key: tuple[str, str]):
        self._validate_key(key)

        if self._FACTORY is None:
            raise NotImplementedError("Subclasses must define a _factory.")
        value = self._FACTORY(name="_".join(key))
        self[key] = value
        return value


class FlowParamsDict(_BaseParamsDict):
    _FACTORY = EquateFlowParams


class InvestParamsDict(_BaseParamsDict):
    _FACTORY = EquateInvestParams


class CustomConstraints:
    def __init__(self, scenario):
        self.scenario = scenario

        self._equal_flows = FlowParamsDict()
        self._equal_invests = InvestParamsDict()
        self._limited_invests = {"flow": [], "storage": []}

    def apply_constraints(self, model):
        # Add pyomo block to model to store custom constraints
        model.CUSTOM_CONSTRAINTS = po.Block()

        # Apply constraints to enforce equal flows
        self._equate_flows(model)

        # Apply additional constraints to equalize investment variables for bidirectional flows
        self._equate_invests(model)

        # Limit initial investment costs
        self._limit_invests(model)

        # Limit energy fed into grids and energy storages for which "res_only" is activated to renewable energies only
        self._limit_to_renewables(model)

    def add_equal_flows(
        self,
        key: tuple[str, str],
        flow1: tuple | None = None,
        flow2: tuple | None = None,
        flows1: list[tuple] | None = None,
        flows2: list[tuple] | None = None,
    ):
        """
        Add one or more flows to a pair of equal-flow groups.

        Each flow is represented as a tuple (from_node, to_node). Flows added to
        `flows1` are constrained to be equal to the corresponding flows in `flows2`
        within the group identified by `key`.

        Parameters
        ----------
        key : tuple[str, str]
            Identifier of the equal-flow group.
        flow1 : tuple, optional
            A single flow to add to the first group.
        flow2 : tuple, optional
            A single flow to add to the second group.
        flows1 : list of tuple, optional
            A list of flows to add to the first group.
        flows2 : list of tuple, optional
            A list of flows to add to the second group.

        Notes
        -----
        If both singular and plural arguments are provided, all flows will be added to their respective groups.
        """

        target_flows1 = self._equal_flows[key].flows1
        target_flows2 = self._equal_flows[key].flows2

        if flow1 is not None:
            target_flows1.append(flow1)
        if flow2 is not None:
            target_flows2.append(flow2)
        if flows1 is not None:
            target_flows1.extend(flows1)
        if flows2 is not None:
            target_flows2.extend(flows2)

    def add_equal_invests(self, key: tuple[str, str], invest: tuple | None = None, invests: list[tuple] | None = None):
        """
        Add one or more investment flows to an equal-investment group.

        An investment flow is represented as a tuple of the form (from_node, to_node).

        Parameters
        ----------
        key : tuple[str, str]
            Identifier of the equal-flow group.
        invest : tuple, optional
            A single investment flow to add.
        invests : list of tuple, optional
            A list of investment flows to add.

        Notes
        -----
        If both `invest` and `invests` are provided, both will be added to the group.
        """
        target = self._equal_invests[key].invests
        if invest is not None:
            target.append(invest)
        if invests is not None:
            target.extend(invests)

    def add_invest_to_limitation(
        self,
        capex_spec: float,
        flow: tuple | None = None,
        storage: solph.components.GenericStorage | None = None,
    ):
        """
        Add an investment object which is to be considered for the investment limit.

        An investment flow is represented as a tuple of the form (from_node, to_node).
        An investment storage is represented by the storage object.

        Parameters
        ----------
        capex_spec : float
            Specific investment cost of the flow or storage to be added.
        flow : tuple, optional
            A single investment flow to add.
        storage : storage object, optional
            A single investment storage to add.

        Notes
        -----
        If both `flow` and `storage` are provided, both will be added using the same `capex_spec` value.
        """

        if flow is not None:
            self._limited_invests["flow"].append({"fi": flow[0], "fo": flow[1], "capex_spec": capex_spec})
        if storage is not None:
            self._limited_invests["storage"].append({"so": storage, "capex_spec": capex_spec})

    def _equate_flows(self, model: solph.Model):
        """
        Forces all flows within an equal flow group stored in self._equal_flows to be identical in the given model
        by using oemof.solph.constraints.equate_flows() to add constraints to the model.

        Parameters
        ----------
        model : solph.Model
            Energy system model
        """
        for v in self._equal_flows.values():
            solph.constraints.equate_flows(model=model, **v.dict)

    def _equate_invests(self, model):
        """
        Forces all flow investments within an equal investment group stored in self._equal_invests to be identical in
        the given model by using oemof.solph.constraints.equate_variables() to add constraints to the model.

        Parameters
        ----------
        model : solph.Model
            Energy system model
        """
        for v in self._equal_invests.values():
            var1 = model.InvestmentFlowBlock.invest[*v.invests[0], 0]  # last 0 -> period ID
            multiple = True if len(v.invests) > 2 else False

            for invest_idx, invest_flow in enumerate(v.invests[1:]):
                solph.constraints.equate_variables(
                    model=model,
                    var1=var1,
                    var2=model.InvestmentFlowBlock.invest[*invest_flow, 0],
                    factor1=1.0,  # has to be one due to multiple unordered investment flows
                    name=f"{v.name}{f'_{invest_idx}' if multiple else ''}",
                )

    def _limit_invests(self, model):
        # Goal:     Limit all initial investment costs to a specified value (neglect peakshaving investments)
        # Approach: Add a constraint adding all initial investment costs and limiting the sum to the specified value
        model.CUSTOM_CONSTRAINTS.LIMIT_INVESTS = po.Block()

        def _limit_invests(m, block, name):
            def _limit_invest_rule(block):
                expr = 0

                # Add investment costs for all flow objects
                expr += sum(
                    m.InvestmentFlowBlock.invest[invest_flow["fi"], invest_flow["fo"], 0] * invest_flow["capex_spec"]
                    for invest_flow in self._limited_invests["flow"]
                )

                # Add investment costs for all storage objects
                expr += sum(
                    m.GenericInvestmentStorageBlock.invest[invest_storage["so"], 0] * invest_storage["capex_spec"]
                    for invest_storage in self._limited_invests["storage"]
                )

                expr += self.scenario.capex_preexisting_considered

                return expr <= self.scenario.invest_max

            setattr(block, name, po.Constraint(rule=_limit_invest_rule))

        # Add additional user-specific constraints for investment cost limit
        if self.scenario.invest_max is not None:
            _limit_invests(m=model, block=model.CUSTOM_CONSTRAINTS.LIMIT_INVESTS, name="limit_invest_costs")

    def _limit_to_renewables(self, model):
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
        for var_name in ["pwr_res_acac", "pwr_res_acdc", "pwr_res_dcac", "pwr_res_dcdc"]:
            setattr(
                model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY, var_name, po.Var(model.TIMEINDEX, within=po.NonNegativeReals)
            )

        # Get discharging flows of all StationaryBattery instances which only allow storing renewable energy
        from_storage_ac = [
            (block.components["outflow"], block.bus_connected)
            for block in self.scenario.block_registry.get("StationaryBattery", {}).values()
            if block.res_only and block.system == "ac"
        ]
        to_storage_ac = [
            (block.bus_connected, block.components["inflow"])
            for block in self.scenario.block_registry.get("StationaryBattery", {}).values()
            if block.res_only and block.system == "ac"
        ]
        from_storage_dc = [
            (block.components["outflow"], block.bus_connected)
            for block in self.scenario.block_registry.get("StationaryBattery", {}).values()
            if block.res_only and block.system == "dc"
        ]
        to_storage_dc = [
            (block.bus_connected, block.components["inflow"])
            for block in self.scenario.block_registry.get("StationaryBattery", {}).values()
            if block.res_only and block.system == "dc"
        ]

        # Get flows of all components connected to each SystemCore bus which only allow feed-in of renewable energy
        flows_res_from_bus = {
            "ac": [
                (market.parent.components["bus"], market.components["snk"])
                for market in self.scenario.block_registry.get("GridMarket", {}).values()
                if market.res_only and market.parent.system == "ac"
            ]
            + to_storage_ac,
            "dc": [
                (market.parent.components["bus"], market.components["snk"])
                for market in self.scenario.block_registry.get("GridMarket", {}).values()
                if market.res_only and market.parent.system == "dc"
            ]
            + to_storage_dc,
        }

        # Get all renewable power flows
        flows_res_to_bus = {
            "ac": [
                (block.components["outflow"], block.bus_connected)
                for block in self.scenario.block_registry.get("RenewableSource", {}).values()
                if block.system == "ac"
            ]
            + from_storage_ac,
            "dc": [
                (block.components["outflow"], block.bus_connected)
                for block in self.scenario.block_registry.get("RenewableSource", {}).values()
                if block.system == "dc"
            ]
            + from_storage_dc,
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
        _sum_res(
            m=model,
            block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
            name="sum_res_ac",
            sum_flow=flows_res_to_bus["ac"],
            split_flows=[
                model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_acac,
                model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_acdc,
            ],
        )
        # define res_dc as sum of pwr_res_dcac and pwr_res_dcdc
        _sum_res(
            m=model,
            block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
            name="sum_res_dc",
            sum_flow=flows_res_to_bus["dc"],
            split_flows=[
                model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcac,
                model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcdc,
            ],
        )

        def _limit_res_to_conv(m, block, name, conv_flow, res_flow):
            def _limit_res2conv_rule(block):
                for p, ts in m.TIMEINDEX:
                    expr = m.flow[conv_flow[0], conv_flow[1], ts] >= res_flow[p, ts]

                    if expr is not True:
                        getattr(block, name).add((p, ts), expr)

            setattr(block, name, po.Constraint(model.TIMEINDEX, noruleinit=True))
            setattr(block, name + "_build", po.BuildAction(rule=_limit_res2conv_rule))

        # limit flow of renewable power from AC to DC to the maximum power of the AC/DC converter in SystemCore
        _limit_res_to_conv(
            m=model,
            block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
            name="limit_pwr_res_acdc_to_conv",
            conv_flow=(
                self.scenario.block_registry.get("TopLevelBlock", {})["core"].components["ac"],
                self.scenario.block_registry.get("TopLevelBlock", {})["core"].components["acdc"],
            ),
            res_flow=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_acdc,
        )
        # limit flow of renewable power from DC to AC to the maximum power of the DC/AC converter in SystemCore
        _limit_res_to_conv(
            m=model,
            block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
            name="limit_pwr_res_dcac_to_conv",
            conv_flow=(
                self.scenario.block_registry.get("TopLevelBlock", {})["core"].components["dc"],
                self.scenario.block_registry.get("TopLevelBlock", {})["core"].components["dcac"],
            ),
            res_flow=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcac,
        )

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
        _limit_feed_in(
            m=model,
            block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
            name="limit_res_ac_feed_in",
            flows_feed_in=flows_res_from_bus["ac"],
            flows_res=[
                model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_acac,
                model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcac,
            ],
            eff_conv=[1, self.scenario.block_registry.get("TopLevelBlock", {})["core"].eff["dcac"]],
        )

        # limit feed-in of renewable power from the DC bus to components connected to the DC-bus considering the
        # SystemCore's converter efficiency
        _limit_feed_in(
            m=model,
            block=model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY,
            name="limit_res_dc_feed_in",
            flows_feed_in=flows_res_from_bus["dc"],
            flows_res=[
                model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcac,
                model.CUSTOM_CONSTRAINTS.RENEWABLES_ONLY.pwr_res_dcdc,
            ],
            eff_conv=[self.scenario.block_registry.get("TopLevelBlock", {})["core"].eff["acdc"], 1],
        )
