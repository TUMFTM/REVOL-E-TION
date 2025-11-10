import collections
from typing import Any, Literal

import numpy as np
import oemof.network
import oemof.solph as solph
from typing_extensions import override

import revoletion.optimization.constraints as constraints
from revoletion import blocks, utils
from revoletion import scenario as scn

_CHARGE_INCENTIVE_MULTIPLIER = -3  # Prioritize storage over curtailment
_WASTE_LOOP_PENALTY_MULTIPLIER = 4  # Prevent inefficient cycling


class WrappedEnergySystem:
    """
    Wrap a `solph.EnergySystem` to allow easy access to its components.

    The oemof optimization results are indexed by each component instance, instead of each component label (like for PyPSA).
    To extract the necessary information after optimization we therefore need a way to map from blocks to individual components.
    Originally, this was achieved with the `components` attribute on each block, but this is not optimization backend agnostic.
    So instead the index is kept inside this wrapper.

    Although components inside an energy system can have unique labels, this oemof API is still experimental
    and can therefore not be used reliably. To work around this limitation, the index is instead built here
    until the oemof API is stable.
    """

    def __init__(self, es: solph.EnergySystem, scenario: scn.Scenario) -> None:
        """
        :param es: The solph energy system which is wrapped.
        :param scenario: The scenario from which the energy system was created. Required for the constraints.
        """
        self.es = es
        self.constraints = constraints.CustomConstraints(scenario)
        self._components = collections.defaultdict(dict)

    def link(self, block: blocks.BaseBlock, label: str, component: oemof.network.Node) -> None:
        """
        Link a oemof component to a block using a label.

        This does not add the component to the energy system. Useful to
        alias different components under one block, e.g., for `bus_connected`.

        :param block: The block to which this component is linked.
        :param label: A unique (in the block scope) label for this component, which can be used to access this component again.
        :param component: The component to be linked.
        """
        if label in self._components[block]:
            raise RuntimeError(
                f"Cannot link component {component} to block {block.name}: A component with the label {label} is already registered for this block"
            )
        self._components[block][label] = component

    def add(self, block: blocks.BaseBlock, label: str, component: oemof.network.Node) -> None:
        """
        Add a new oemof component to the energy system and name index.

        This also adds the component to the legacy component index inside each block.

        :param block: The block from which this component was created.
        :param label: A unique (in the block scope) label for this component, which can be used to access this component again.
        :param component: The component to be added.
        """
        # Add the component to the internal index.
        self.link(block, label, component)
        # Add the component to the energy system so it appears in the oemof model.
        self.es.add(component)

        # The `revoletion.optimization.constraints.CustomConstraints` still require access to the oemof components through
        # the legacy blocks component index.
        # TODO: remove this once `revoletion.optimization.constraints` has been reworked.
        if hasattr(block, "components"):
            # Only add the component, if the block has a components index.
            block.components[label] = component

    def get_component(self, block: blocks.BaseBlock, label: str) -> oemof.network.Node:
        """
        Get the component of a block using its label.
        """
        if label not in self._components[block]:
            print(self._components[block])
            raise ValueError(
                f"Cannot get component '{label}' for block '{block.name}': component is not registered for block"
            )
        return self._components[block][label]

    def get_components(self, block: blocks.BaseBlock) -> dict[str, oemof.network.Node]:
        """
        Get all components for a block.

        Useful if the number of components is not known, e.g., for `GridConnection`.
        """
        return self._components[block]


class OemofBlockVisitor(blocks.BlockVisitor[None]):
    """
    Visitor to construct an oemof optimization model from a REVOL-E-TION scenario.

    This visitor traverses the hierarchical block structure of an energy system and creates
    corresponding oemof.solph components.
    """

    # Since each top level block connects to either the AC or DC bus, but they are not children
    # of the `SystemCore` block, the names for the core buses are fixed.
    # This way, each block can be easily assigned to the AC or DC core bus.
    _CORE_AC_BUS_NAME = "ac"
    _CORE_DC_BUS_NAME = "dc"

    def __init__(self, scenario: scn.Scenario, horizon: utils.TimeSettings, cost_eps: float) -> None:
        self._scenario = scenario
        self._horizon = horizon
        self._cost_eps = cost_eps

    @classmethod
    def create_oemof_energy_system(
        cls, scenario: scn.Scenario, horizon: utils.TimeSettings, cost_eps: float
    ) -> WrappedEnergySystem:
        """
        Factory method that constructs a complete oemof energy system from a scenario.

        :param scenario: Scenario with all blocks and parameters.
        :param horizon: Time settings for the optimization period.
        :param cost_eps: Small positive value for numerical tie-breaking.
        :return: A fully constructed wrapped energy system ready for optimization.
        """
        es = solph.EnergySystem(timeindex=horizon.dti, infer_last_interval=True)
        wrapped_es = WrappedEnergySystem(es, scenario)
        visitor = cls(scenario, horizon, cost_eps)

        # Logically the root node of the block structure is the system core, since any node is
        # connected to either its AC or DC bus. The oemof components of the core are therefore
        # created before any other block is processed, so no block tries to reference undefined buses.
        system_core_block = scenario.block_registry["TopLevelBlock"]["core"]
        visitor.visit_system_core(system_core_block, es=wrapped_es)

        for block in scenario.block_registry.get("TopLevelBlock", {}).values():
            visitor.visit_block(block, es=wrapped_es)
        return wrapped_es

    @override
    def visit_block(
        self, block: blocks.BaseBlock, es: WrappedEnergySystem, bus_connected: solph.Bus | None = None
    ) -> None:
        """
        Dispatch visitor to the appropriate visit method based on block type.

        :param block: The block to be converted to oemof components.
        :param es: The wrapped energy system to add components to.
        :param bus_connected: An optional bus, to which the block is connected. If not given, it is determined from the blocks system (i.e., AC or DC core bus).

        :raises RuntimeError: If an unsupported block type is encountered.
        """

        # If the block has a specified system (i.e., "AC" or "DC") but no given bus it should
        # be connected to the relevant core bus.
        # This is usually the case for all top level blocks.
        if hasattr(block, "system") and bus_connected is None:
            bus_connected = self._get_core_bus(block.system, es)

        if bus_connected is not None:
            # If a bus is given or was previously determined, it should be linked, so the result processing can later access it easily.
            es.link(block, "bus-connected", bus_connected)
            # Also add the bus to the legacy `bus_connected` attribute for the constraints handling.
            block.bus_connected = bus_connected

        match block:
            case blocks.SystemCore():
                return
            case blocks.GridConnection():
                return self.visit_grid_connection(block, es=es, bus_connected=bus_connected)
            case blocks.GridMarket():
                return self.visit_grid_market(block, es=es, bus_connected=bus_connected)
            case blocks.RenewableSource():
                return self.visit_renewable_source(block, es=es, bus_connected=bus_connected)
            case blocks.ControllableSource():
                return self.visit_controllable_source(block, es=es, bus_connected=bus_connected)
            case blocks.FixedDemand():
                return self.visit_fixed_demand(block, es=es, bus_connected=bus_connected)
            case blocks.StationaryBattery():
                return self.visit_stationary_battery(block, es=es, bus_connected=bus_connected)
            case blocks.Fleet():
                return self.visit_fleet(block, es=es, bus_connected=bus_connected)
            case blocks.SubFleet():
                return self.visit_sub_fleet(block, es=es, bus_connected=bus_connected)
            case blocks.ElectricFleetUnit():
                return self.visit_electric_fleet_unit(block, es=es, bus_connected=bus_connected)
            case _:
                raise RuntimeError(f"Cannot visit block {block.name}: block of type {type(block)} is not supported")

    def _get_core_bus(self, system: Literal["ac", "dc"], es: WrappedEnergySystem) -> solph.Bus:
        system_core_block = self._scenario.block_registry["TopLevelBlock"]["core"]
        if system == "ac":
            return es.get_component(system_core_block, self._CORE_AC_BUS_NAME)
        else:
            return es.get_component(system_core_block, self._CORE_DC_BUS_NAME)

    def visit_system_core(self, block: blocks.SystemCore, es: WrappedEnergySystem) -> None:
        """
        Create the bidirectional AC-DC conversion infrastructure at the system's core.

        x denotes the flow measurement point in results

        dc          ac
        |-x--dcac-->|
        |           |
        |<---acdc-x-|
        """
        ac_bus = solph.Bus(label=self._CORE_AC_BUS_NAME)
        dc_bus = solph.Bus(label=self._CORE_DC_BUS_NAME)

        acdc_converter = solph.components.Converter(
            label=f"{block.name}-acdc",
            inputs={
                ac_bus: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=block.evaluators["acdc"].opt.spec_ep_invest,
                        existing=block.sizes["acdc"].preexisting,
                        maximum=block.sizes["acdc"].expansion_max,
                    ),
                    variable_costs=block.evaluators["acdc"].opt.spec_ep_operation[self._horizon.dti],
                )
            },
            outputs={dc_bus: solph.Flow(variable_costs=self._cost_eps)},
            conversion_factors={dc_bus: block.eff["acdc"]},
        )

        dcac_converter = solph.components.Converter(
            label=f"{block.name}-dcac",
            inputs={
                dc_bus: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=block.evaluators["dcac"].opt.spec_ep_invest,
                        existing=block.sizes["dcac"].preexisting,
                        maximum=block.sizes["dcac"].expansion_max,
                    ),
                    variable_costs=block.evaluators["dcac"].opt.spec_ep_operation[self._horizon.dti],
                )
            },
            outputs={ac_bus: solph.Flow(variable_costs=self._cost_eps)},
            conversion_factors={ac_bus: block.eff["dcac"]},
        )

        es.add(block, self._CORE_AC_BUS_NAME, ac_bus)
        es.add(block, self._CORE_DC_BUS_NAME, dc_bus)
        es.add(block, "acdc", acdc_converter)
        es.add(block, "dcac", dcac_converter)

        es.constraints.add_invest_costs(
            invest=(ac_bus, acdc_converter),
            capex_spec=block.evaluators["acdc"].capex.spec,
            invest_type="flow",
        )

        es.constraints.add_invest_costs(
            invest=(dc_bus, dcac_converter),
            capex_spec=block.evaluators["dcac"].capex.spec,
            invest_type="flow",
        )

        if block.expansion_equal:
            # add a tuple of tuples to the list of equal variables of the scenario
            es.constraints.add_equal_invests(
                [
                    {"in": dc_bus, "out": dcac_converter},
                    {"in": ac_bus, "out": acdc_converter},
                ]
            )

    def visit_renewable_source(
        self, block: blocks.RenewableSource, es: WrappedEnergySystem, bus_connected: solph.Bus
    ) -> None:
        """
        Build oemof model for a `RenewableSource`.

        bus_connected      name_bus
          |                   |
          |<--x----name_out---|<--name_src
          |                   |
          |                   |-->name_exc
        """

        bus_internal = solph.Bus(label=f"{block.name}-bus")

        outflow_converter = solph.components.Converter(
            label=_label(block, "outflow"),
            inputs={bus_internal: solph.Flow()},
            outputs={bus_connected: solph.Flow()},
            conversion_factors={bus_connected: block.eff["block"]},
        )

        exc = solph.components.Sink(label=_label(block, "exc"), inputs={bus_internal: solph.Flow()})

        src = solph.components.Source(
            label=_label(block, "src"),
            outputs={
                bus_internal: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=block.evaluators["block"].opt.spec_ep_invest,
                        existing=block.sizes["block"].preexisting,
                        maximum=block.sizes["block"].expansion_max,
                    ),
                    fix=block.data.loc[self._horizon.dti, "power_spec"],
                    variable_costs=block.evaluators["block"].opt.spec_ep_operation[self._horizon.dti],
                )
            },
        )

        es.add(block, "bus", bus_internal)
        es.add(block, "outflow", outflow_converter)
        es.add(block, "exc", exc)
        es.add(block, "src", src)

        es.constraints.add_invest_costs(
            invest=(src, bus_internal),
            capex_spec=block.evaluators["block"].capex.spec,
            invest_type="flow",
        )

    def visit_fixed_demand(self, block: blocks.FixedDemand, es: WrappedEnergySystem, bus_connected: solph.Bus) -> None:
        """
        Build oemof model for a `FixedDemand`.

        x denotes the flow measurement point in results

        bus_connected
          |
          |-x->name_snk
          |
        """

        sink = solph.components.Sink(
            label=_label(block, "snk"),
            inputs={
                bus_connected: solph.Flow(nominal_capacity=1, fix=block.flows_apriori["demand"][self._horizon.dti])
            },
        )
        es.add(block, "snk", sink)

    def visit_controllable_source(
        self, block: blocks.ControllableSource, es: WrappedEnergySystem, bus_connected: solph.Bus
    ) -> None:
        """
        Build oemof model for a `ControllableSource`.

        x denotes the flow measurement point in results

        bus_connected
          |
          |<-name_gen
          |
        """

        src = solph.components.Source(
            label=_label(block, "src"),
            outputs={
                bus_connected: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=block.evaluators["block"].opt.spec_ep_invest,
                        existing=block.sizes["block"].preexisting,
                        maximum=block.sizes["block"].expansion_max,
                    ),
                    variable_costs=block.evaluators["block"].opt.spec_ep_operation[self._horizon.dti],
                )
            },
        )
        es.add(block, "src", src)
        es.constraints.add_invest_costs(
            invest=(src, bus_connected),
            capex_spec=block.evaluators["block"].capex.spec,
            invest_type="flow",
        )

    def visit_grid_connection(
        self, block: blocks.GridConnection, es: WrappedEnergySystem, bus_connected: solph.Bus
    ) -> None:
        """
        Build oemof model for a `GridConnection`.

        x denotes the flow measurement point in results

        bus_connected          name_bus
          |                        |
          |---name_inflow_1--x---->|
          |<--name_outflow_1--x----|
          |                        |---(GridMarket Instance)
          |---name_inflow_2--x---->|
          |<--name_outflow_2--x----|
          |                        |---(GridMarket Instance)

                     ...

          |---name_inflow_n--x---->|
          |<--name_outflow_n--x----|
        """

        bus_internal = solph.Bus(label=_label(block, "bus"))
        es.add(block, "bus", bus_internal)

        # Inflow (Grid -> System)
        inflow_1 = solph.components.Converter(
            label=_label(block, "inflow_1"),
            # Peakshaving not implemented for feed-in into grid
            inputs={bus_connected: solph.Flow()},
            # Size optimization
            outputs={
                bus_internal: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=block.evaluators["s2g"].opt.spec_ep_invest,
                        existing=block.sizes["s2g"].preexisting,
                        maximum=block.sizes["s2g"].expansion_max,
                    ),
                    variable_costs=self._cost_eps,
                )
            },
            conversion_factors={bus_internal: 1},
        )
        es.add(block, "inflow_1", inflow_1)
        # For compatibility with `CustomConstraints` the inflows must be saved on the block.
        block.inflows = {f"{block.name}_inflow_1": inflow_1}

        # Outflows
        outflows = {}
        equal_investments = []

        first_period = block.peak_periods.index[0]

        for period in block.peak_periods.index:
            assign_invest_costs = period == first_period
            outflow = solph.components.Converter(
                label=_label(block, f"outflow_{period}"),
                # Size optimization: investment costs are assigned to first peakshaving interval only. The application of
                # constraints ensures that the optimized grid connection sizes of all peakshaving intervals are equal
                inputs={
                    bus_internal: solph.Flow(
                        nominal_capacity=solph.Investment(
                            ep_costs=(block.evaluators["g2s"].opt.spec_ep_invest if assign_invest_costs else 0),
                            existing=block.sizes["g2s"].preexisting,
                            maximum=block.sizes["g2s"].expansion_max,
                        )
                    )
                },
                # Peakshaving
                outputs={
                    bus_connected: solph.Flow(
                        nominal_capacity=(
                            solph.Investment(
                                ep_costs=(block.evaluators[period].opt.spec_ep_peak if block.peakshaving else 0),
                                existing=block.peak_periods.loc[period, "power"],
                            )
                        ),
                        max=(block.bus_activation.loc[self._horizon.dti, period]),
                    )
                },
                conversion_factors={bus_connected: 1},
            )

            es.add(block, f"outflow_{period}", outflow)
            outflows[f"{block.name}_outflow_{period}"] = outflow

            if assign_invest_costs:
                es.constraints.add_invest_costs(
                    invest=(
                        bus_internal,
                        outflow,
                    ),
                    capex_spec=block.evaluators["g2s"].capex.spec,
                    invest_type="flow",
                )

            equal_investments.append({"in": bus_internal, "out": outflow})

        # For compatibility with `CustomConstraints` the outflows must be saved on the block.
        block.outflows = outflows

        # If size of in- and outflow from and to the grid have to be the same size, add outflow investment(s)
        if block.expansion_equal:
            equal_investments.append(
                {
                    "in": inflow_1,
                    "out": bus_internal,
                }
            )  # currently only works without peakshaving for inflows

        # add list of variables to the scenario constraints if list contains more than one element
        # lists with one element occur, if peakshaving is deactivated and grid sizes don't have to be equal
        if len(equal_investments) > 1:
            es.constraints.add_equal_invests(equal_investments)

        es.constraints.add_invest_costs(
            invest=(inflow_1, bus_internal),
            capex_spec=block.evaluators["s2g"].capex.spec,
            invest_type="flow",
        )

        for subblock in block.subblocks.values():
            if not isinstance(subblock, blocks.GridMarket):
                raise ValueError(
                    f"Expected each grid connection subblock to be of type {type(blocks.GridMarket)} but got {type(subblock)}"
                )

            self.visit_block(subblock, es, bus_connected=bus_internal)

    def visit_grid_market(self, block: blocks.GridMarket, es: WrappedEnergySystem, bus_connected: solph.Bus) -> None:
        """
        Build oemof model for a `GridMarket`.

        parent_bus
            |<---x----name_src
            |
            |----x--->name_snk
            |
        """
        src = solph.components.Source(
            label=_label(block, "src"),
            outputs={
                bus_connected: solph.Flow(
                    nominal_capacity=block.pwr_g2s,
                    max=1 if block.pwr_g2s else None,
                    variable_costs=block.evaluators["g2s"].opt.spec_ep_operation[self._horizon.dti],
                )
            },
        )

        snk = solph.components.Sink(
            label=_label(block, "snk"),
            inputs={
                bus_connected: solph.Flow(
                    nominal_capacity=block.pwr_s2g,
                    max=1 if block.pwr_s2g else None,
                    variable_costs=block.evaluators["s2g"].opt.spec_ep_operation[self._horizon.dti],
                )
            },
        )

        es.add(block, "src", src)
        es.add(block, "snk", snk)

    def visit_stationary_battery(
        self, block: blocks.StationaryBattery, es: WrappedEnergySystem, bus_connected: solph.Bus
    ) -> None:
        """
        Build oemof model for a `StationaryBattery`.

        x denotes the flow measurement point in results

        bus_connected   name_bus
             |             |
             |<-x-name_xc--|
             |             |<--->name_ess
             |-x-name_ess->|
             |             |
        """

        params = {
            "inflow_nominal_capacity": None,
            "outflow_nominal_capacity": None,
            "inflow_max": None,
            "outflow_max": None,
            "inflow_fix": None,
            "outflow_fix": None,
            "invest_relation_input_capacity": block.crate_chg,
            "invest_relation_output_capacity": block.crate_dis,
            "storage_balanced": True if self._scenario.strategy == "go" else False,
        }
        self._visit_storage_block(block, es, bus_connected, params)

    def visit_fleet(self, block: blocks.Fleet, es: WrappedEnergySystem, bus_connected: solph.Bus) -> None:
        """
        Build oemof model for a `Fleet`.

        x denotes the flow measurement point in results
        xc denotes ac or dc, depending on the parameter 'system'

        bus_connected        name_bus
          |<----name_outflow--x-|---(ElectricFleetUnit Instance)
          |                     |
          |-x----name_inflow--->|---(ElectricFleetUnit Instance)
          |                     |
          |                     |   (CombustionVehicle Instance)
        """

        bus = solph.Bus(label=_label(block, "bus"))

        inflow = solph.components.Converter(
            label=_label(block, "inflow"),
            inputs={
                bus_connected: solph.Flow(
                    variable_costs=block.evaluators["s2f"].opt.spec_ep_operation[self._horizon.dti],
                    nominal_capacity=block.pwr_lim_s2f,
                    # default value for max is 1; not explicitly set to ensure compatibility with nominal_capacity=None
                )
            },
            outputs={bus: solph.Flow()},
            conversion_factors={bus: 1},
        )

        outflow = solph.components.Converter(
            label=_label(block, "outflow"),
            inputs={
                bus: solph.Flow(
                    variable_costs=block.evaluators["f2s"].opt.spec_ep_operation[self._horizon.dti],
                    nominal_capacity=block.pwr_lim_f2s,
                    # default value for max is 1; not explicitly set to ensure compatibility with nominal_capacity=None
                )
            },
            outputs={bus_connected: solph.Flow(variable_costs=self._cost_eps)},
            conversion_factors={block.bus_connected: 1},
        )
        es.add(block, "bus", bus)
        es.add(block, "inflow", inflow)
        es.add(block, "outflow", outflow)

        for subblock in block.subblocks.values():
            if not isinstance(subblock, blocks.SubFleet):
                raise ValueError(
                    f"Expected each fleet subblock to be of type {type(blocks.SubFleet)} but got {type(subblock)}"
                )

            self.visit_block(subblock, es, bus_connected=bus)

    def visit_sub_fleet(self, block: blocks.SubFleet, es: WrappedEnergySystem, bus_connected: solph.Bus) -> None:
        for subblock in block.subblocks.values():
            if isinstance(subblock, blocks.ElectricFleetUnit):
                self.visit_block(subblock, es, bus_connected)

    def visit_electric_fleet_unit(
        self, block: blocks.ElectricFleetUnit, es: WrappedEnergySystem, bus_connected: solph.Bus
    ) -> None:
        """
        Build oemof model for an `ElectricFleetUnit`.

        parent.parent_bus     name_bus
            |<--x--name_fleet---|<-x->name_storage (handled in `_visit_storage_block`)
            |                   |
            |---x--fleet_name-->|-->name_snk (handled in `_visit_storage_block`)
            |                   |
            |                   |<--name_ext_ac-x- (external charging AC)
            |                   |
            |                   |<--name_ext_dc-x- (external charging DC)

        :param block: The `ElectricFleetUnit` block to process.
        :param es: The energy system to which the components of the block are added.
        :param bus_connected: The `Fleet` bus, the `ElectricFleetUnit` should be attached to.
        """

        params = {
            "inflow_nominal_capacity": block.pwr_chg_max,
            "outflow_nominal_capacity": block.pwr_dis_max * block.eff["dis_int"],
            "inflow_max": None if block.apriori else block.log.loc[self._horizon.dti, "atbase"].astype(int),
            "outflow_max": None if block.apriori else block.log.loc[self._horizon.dti, "atbase"].astype(int),
            "inflow_fix": block.flows_apriori.loc[self._horizon.dti, "p_int_chg"] if block.apriori else None,
            "outflow_fix": block.flows_apriori.loc[self._horizon.dti, "p_int_dis"] if block.apriori else None,
            "invest_relation_input_capacity": None,
            "invest_relation_output_capacity": None,
            "storage_balanced": False,
        }
        bus_internal = self._visit_storage_block(block, es, bus_connected, params)

        snk = solph.components.Sink(
            label=_label(block, "snk"),
            inputs={bus_internal: solph.Flow(nominal_capacity=1, fix=block.log.loc[self._horizon.dti, "consumption"])},
        )

        bus_ext_ac = solph.Bus(label=_label(block, "bus_ext_ac"))

        src_ext_ac = solph.components.Source(
            label=_label(block, "src_ext_ac"),
            outputs={
                bus_ext_ac: solph.Flow(
                    nominal_capacity=block.pwr_ext_ac_max,
                    max=None if block.apriori else block.log.loc[self._horizon.dti, "atac"].astype(int),
                    fix=block.flows_apriori.loc[self._horizon.dti, "p_ext_ac_chg"] if block.apriori else None,
                    variable_costs=block.evaluators["ext_ac"].opt.spec_ep_operation[self._horizon.dti],
                )
            },
        )

        conv_ext_ac = solph.components.Converter(
            label=_label(block, "conv_ext_ac"),
            inputs={bus_ext_ac: solph.Flow()},
            outputs={bus_internal: solph.Flow()},
            conversion_factors={bus_internal: block.eff["chg_ac"]},
        )

        bus_ext_dc = solph.Bus(label=_label(block, "bus_ext_dc"))

        src_ext_dc = solph.components.Source(
            label=_label(block, "src_ext_dc"),
            outputs={
                bus_ext_dc: solph.Flow(
                    nominal_capacity=block.pwr_ext_dc_max,
                    max=None if block.apriori else block.log.loc[self._horizon.dti, "atdc"].astype(int),
                    fix=block.flows_apriori.loc[self._horizon.dti, "p_ext_dc_chg"] if block.apriori else None,
                    variable_costs=block.evaluators["ext_dc"].opt.spec_ep_operation[self._horizon.dti],
                )
            },
        )

        conv_ext_dc = solph.components.Converter(
            label=_label(block, "conv_ext_dc"),
            inputs={bus_ext_dc: solph.Flow()},
            outputs={bus_internal: solph.Flow()},
            conversion_factors={bus_internal: 1},  # billed energy is already dc in external dc charging
        )

        es.add(block, "snk", snk)

        # External AC charging.
        es.add(block, "bus_ext_ac", bus_ext_ac)
        es.add(block, "src_ext_ac", src_ext_ac)
        es.add(block, "conv_ext_ac", conv_ext_ac)

        # External DC charging.
        es.add(block, "bus_ext_dc", bus_ext_dc)
        es.add(block, "src_ext_dc", src_ext_dc)
        es.add(block, "conv_ext_dc", conv_ext_dc)

    def _visit_storage_block(
        self, block: blocks.StorageBlock, es: WrappedEnergySystem, bus_connected: solph.Bus, params: dict[str, Any]
    ) -> solph.Bus:
        """
        Generic handler to build the oemof models for `StorageBlock`s.

        :param block: The `StorageBlock`.
        :param es: The energy system to which the components are added.
        :param bus_connected: The parent bus, to which the `StorageBlock` is connected.
        :param params: Parameters for the storage.

        :returns: The internal bus of the storage block, so callers can attach custom components to the storage.
        """

        bus_internal = solph.Bus(label=_label(block, "bus"))

        inflow = solph.components.Converter(
            label=_label(block, "inflow"),
            inputs={
                bus_connected: solph.Flow(
                    nominal_capacity=params["inflow_nominal_capacity"],
                    max=params["inflow_max"],
                    fix=params["inflow_fix"],
                )
            },
            outputs={
                bus_internal: solph.Flow(
                    variable_costs=self._cost_eps
                    * _CHARGE_INCENTIVE_MULTIPLIER  # incentivize charging of StorageBlocks vs. curtailment
                )
            },
            conversion_factors={bus_internal: block.eff["chg_int"]},
        )

        outflow = solph.components.Converter(
            label=_label(block, "outflow"),
            inputs={bus_internal: solph.Flow()},
            outputs={
                bus_connected: solph.Flow(
                    nominal_capacity=params["outflow_nominal_capacity"],
                    max=params["outflow_max"],
                    fix=params["outflow_fix"],
                    variable_costs=self._cost_eps
                    * _WASTE_LOOP_PENALTY_MULTIPLIER,  # disincentivize waste loop with inflow (sum must be positive)
                )
            },
            conversion_factors={bus_connected: block.eff["dis_int"]},
        )

        storage = solph.components.GenericStorage(
            label=_label(block, "storage"),
            inputs={
                bus_internal: solph.Flow(variable_costs=block.evaluators["in"].opt.spec_ep_operation[self._horizon.dti])
            },
            outputs={bus_internal: solph.Flow(variable_costs=self._cost_eps)},
            loss_rate=block.loss_rate_per_hour,
            balanced=params["storage_balanced"],
            initial_storage_level=block.states.loc[self._horizon.start, ["soc", "soc_min", "soc_max"]].median(),
            # crate measured "outside" of conversion factor (efficiency)
            invest_relation_input_capacity=params["invest_relation_input_capacity"],
            invest_relation_output_capacity=params["invest_relation_output_capacity"],
            inflow_conversion_factor=np.sqrt(block.eff["storage_roundtrip"]),
            outflow_conversion_factor=np.sqrt(block.eff["storage_roundtrip"]),
            nominal_capacity=solph.Investment(
                ep_costs=block.evaluators["storage"].opt.spec_ep_invest,
                existing=block.sizes["storage"].preexisting,
                maximum=block.sizes["storage"].expansion_max,
            ),
            max_storage_level=block.states.loc[self._horizon.dti_extd, "soc_max"],
            min_storage_level=block.states.loc[self._horizon.dti_extd, "soc_min"],
        )

        es.add(block, "bus", bus_internal)
        es.add(block, "inflow", inflow)
        es.add(block, "outflow", outflow)
        es.add(block, "storage", storage)

        es.constraints.add_invest_costs(
            invest=(storage,),
            capex_spec=block.evaluators["storage"].capex.spec,
            invest_type="storage",
        )

        return bus_internal


def _label(block: blocks.BaseBlock, label: str) -> str:
    return f"{block.name}-{label}"
