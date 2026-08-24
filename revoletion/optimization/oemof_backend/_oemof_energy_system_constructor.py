import collections
import logging
from typing import Any, Literal, override

import numpy as np
import oemof.network
import oemof.solph as solph
import pandas as pd

import revoletion.optimization.constraints as constraints
from revoletion import blocks, time
from revoletion import scenario as scn

_CHARGE_INCENTIVE_MULTIPLIER = -3  # Prioritize storage over curtailment
_WASTE_LOOP_PENALTY_MULTIPLIER = 4  # Prevent inefficient cycling


class OemofEnergySystemContext:
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

    def __init__(self, dti: pd.DatetimeIndex, scenario: scn.Scenario) -> None:
        """
        :param dti: DatetimeIndex for the energy system.
        :param scenario: The scenario from which the energy system was created. Required for the constraints.
        """
        self._energy_system = solph.EnergySystem(timeindex=dti, infer_last_interval=True)
        self._constraints = constraints.CustomConstraints(scenario)
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
        self._energy_system.add(component)

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

    def build_oemof_model(self, debug: bool = False) -> solph.Model:
        model = solph.Model(self._energy_system, debug=debug)

        self._constraints.apply_constraints(model=model)

        return model


class OemofEnergySystemConstructor(blocks.BlockVisitor[None]):
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

    def __init__(
        self,
        scenario: scn.Scenario,
        horizon: time.TimeFrame,
        cost_eps: float,
        logger: logging.Logger,
        storage_reward_eps: float = 0.0,
    ) -> None:
        self._scenario = scenario
        self._horizon = horizon
        self._cost_eps = cost_eps
        self._storage_reward_eps = storage_reward_eps
        self._logger = logger

    @classmethod
    def create_oemof_energy_system(
        cls,
        scenario: scn.Scenario,
        horizon: time.TimeFrame,
        cost_eps: float,
        logger: logging.Logger,
        storage_reward_eps: float = 0.0,
    ) -> OemofEnergySystemContext:
        """
        Factory method that constructs a complete oemof energy system from a scenario.

        :param scenario: Scenario with all blocks and parameters.
        :param horizon: Time settings for the optimization period.
        :param cost_eps: Small positive value for numerical tie-breaking.
        :param storage_reward_eps: Small positive magnitude rewarding stored energy,
            negated at the call site, see
            OptimizationProblemConfig.storage_reward_eps.
        :return: A fully constructed wrapped energy system ready for optimization.
        """
        wrapped_es = OemofEnergySystemContext(horizon.dti, scenario)
        visitor = cls(scenario, horizon, cost_eps, logger, storage_reward_eps)

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
        self, block: blocks.BaseBlock, es: OemofEnergySystemContext, bus_connected: solph.Bus | None = None
    ) -> None:
        """
        Dispatch visitor to the appropriate visit method based on block type.

        :param block: The block to be converted to oemof components.
        :param es: The wrapped energy system to add components to.
        :param bus_connected: An optional bus, to which the block is connected. If not given, it is determined from the blocks system (i.e., AC or DC core bus).

        :raises RuntimeError: If an unsupported block type is encountered.
        """
        # Try to determine the bus the block should be connected to.
        # If the block has a specified system (i.e., "AC" or "DC") but no given bus it should
        # be connected to the relevant core bus.
        # This is usually the case for all top level blocks.
        if bus_connected is None:
            block_system = getattr(block, "system", None)
            if block_system is None:
                self._logger.debug(f"Block {block.name} skipped since no connected bus could be determined")
                return

            bus_connected = self._get_core_bus(block_system, es)

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

    def _get_core_bus(self, system: Literal["ac", "dc"], es: OemofEnergySystemContext) -> solph.Bus:
        system_core_block = self._scenario.block_registry["TopLevelBlock"]["core"]
        if system == "ac":
            bus = es.get_component(system_core_block, self._CORE_AC_BUS_NAME)
        else:
            bus = es.get_component(system_core_block, self._CORE_DC_BUS_NAME)

        if not isinstance(bus, solph.Bus):
            raise RuntimeError(f"Unexpected oemof node type for {system} bus: {type(bus)}")

        return bus

    def visit_system_core(self, block: blocks.SystemCore, es: OemofEnergySystemContext) -> None:
        """
        Create the bidirectional AC-DC conversion infrastructure at the system's core.

        x denotes the flow measurement point in results

                    dc          ac
         deficit_dc->|-x--dcac-->|<-x-deficit_ac
                    |           |
                    |<---acdc-x-|
        """
        ac_bus = solph.Bus(label=self._CORE_AC_BUS_NAME)
        es.add(block, self._CORE_AC_BUS_NAME, ac_bus)

        dc_bus = solph.Bus(label=self._CORE_DC_BUS_NAME)
        es.add(block, self._CORE_DC_BUS_NAME, dc_bus)

        # The deficit sources are unlimited in power and keep the energy system solvable even if no other component
        # can cover the demand. Their high specific opex makes them the optimizer's last resort.
        for system, bus in ((self._CORE_AC_BUS_NAME, ac_bus), (self._CORE_DC_BUS_NAME, dc_bus)):
            deficit_source = solph.components.Source(
                label=_label(block, f"deficit_{system}"),
                outputs={
                    bus: solph.Flow(variable_costs=block.pois[f"deficit_{system}"].spec_ep_operation[self._horizon.dti])
                },
            )
            es.add(block, f"deficit_{system}", deficit_source)

        acdc_converter = solph.components.Converter(
            label=_label(block, "acdc"),
            inputs={
                ac_bus: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=block.pois["acdc"].spec_ep_invest,
                        existing=block.sizes["acdc"].preexisting,
                        maximum=block.sizes["acdc"].expansion_max,
                    ),
                    variable_costs=block.pois["acdc"].spec_ep_operation[self._horizon.dti],
                )
            },
            outputs={dc_bus: solph.Flow(variable_costs=self._cost_eps)},
            conversion_factors={dc_bus: block.eff["acdc"]},
        )
        es.add(block, "acdc", acdc_converter)

        dcac_converter = solph.components.Converter(
            label=_label(block, "dcac"),
            inputs={
                dc_bus: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=block.pois["dcac"].spec_ep_invest,
                        existing=block.sizes["dcac"].preexisting,
                        maximum=block.sizes["dcac"].expansion_max,
                    ),
                    variable_costs=block.pois["dcac"].spec_ep_operation[self._horizon.dti],
                )
            },
            outputs={ac_bus: solph.Flow(variable_costs=self._cost_eps)},
            conversion_factors={ac_bus: block.eff["dcac"]},
        )
        es.add(block, "dcac", dcac_converter)

        es._constraints.add_invest_to_limitation(
            flow=(ac_bus, acdc_converter),
            capex_spec=block.pois["acdc"].capex.spec,
        )

        es._constraints.add_invest_to_limitation(
            flow=(dc_bus, dcac_converter),
            capex_spec=block.pois["dcac"].capex.spec,
        )

        if block.expansion_equal:
            # add a tuple of tuples to the list of equal variables of the scenario
            es._constraints.add_equal_invests(
                key=(block.name, "invest"),
                invests=[
                    (dc_bus, dcac_converter),
                    (ac_bus, acdc_converter),
                ],
            )

    def visit_renewable_source(
        self, block: blocks.RenewableSource, es: OemofEnergySystemContext, bus_connected: solph.Bus
    ) -> None:
        """
        Build oemof model for a `RenewableSource`.

        bus_connected      name_bus
          |                   |
          |<--x----name_out---|<--name_src
          |                   |
          |                   |-->name_exc
        """

        bus_internal = solph.Bus(label=_label(block, "bus"))
        es.add(block, "bus", bus_internal)

        outflow_converter = solph.components.Converter(
            label=_label(block, "outflow"),
            inputs={bus_internal: solph.Flow()},
            outputs={bus_connected: solph.Flow()},
            conversion_factors={bus_connected: block.eff["block"]},
        )
        es.add(block, "outflow", outflow_converter)

        # Curtailment has to be disincentivized in the optimization to force optimizer to charge storage or commodities
        # instead of curtailment. 2x cost_eps is required as SystemCore also has ccost_eps in charging direction.
        # All other components such as converters and storages only have cost_eps in the output direction.
        exc = solph.components.Sink(label=_label(block, "exc"), inputs={bus_internal: solph.Flow()})
        es.add(block, "exc", exc)

        src = solph.components.Source(
            label=_label(block, "src"),
            outputs={
                bus_internal: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=block.pois["block"].spec_ep_invest,
                        existing=block.sizes["block"].preexisting,
                        maximum=block.sizes["block"].expansion_max,
                    ),
                    fix=block.data.loc[self._horizon.dti, "power_spec"],
                    variable_costs=block.pois["block"].spec_ep_operation[self._horizon.dti],
                )
            },
        )
        es.add(block, "src", src)

        es._constraints.add_invest_to_limitation(
            flow=(src, bus_internal),
            capex_spec=block.pois["block"].capex.spec,
        )

    def visit_fixed_demand(
        self, block: blocks.FixedDemand, es: OemofEnergySystemContext, bus_connected: solph.Bus
    ) -> None:
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
        self, block: blocks.ControllableSource, es: OemofEnergySystemContext, bus_connected: solph.Bus
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
                        ep_costs=block.pois["block"].spec_ep_invest,
                        existing=block.sizes["block"].preexisting,
                        maximum=block.sizes["block"].expansion_max,
                    ),
                    variable_costs=block.pois["block"].spec_ep_operation[self._horizon.dti],
                )
            },
        )
        es.add(block, "src", src)
        es._constraints.add_invest_to_limitation(
            flow=(src, bus_connected),
            capex_spec=block.pois["block"].capex.spec,
        )

    def visit_grid_connection(
        self, block: blocks.GridConnection, es: OemofEnergySystemContext, bus_connected: solph.Bus
    ) -> None:
        """
        Build oemof model for a `GridConnection`.

        x denotes the flow measurement point in results

        bus_connected          name_bus
          |                        |
          |---name_inflow----x---->|---(GridMarket Instance)
          |<--name_outflow----x----|          ...
          |                        |
          |                        |---(GridMarket Instance)


               peak_bus_in              peak_bus_out
                    |                        |
                    |--->storage_period_1--->|
        peak_src--->|           ...          |--->peak_snk
                    |                        |
                    |--->storage_period_n--->|
                    |                        |
        """

        bus_internal = solph.Bus(label=_label(block, "bus"))
        es.add(block, "bus", bus_internal)

        inflow_converter = solph.components.Converter(
            label=_label(block, "inflow"),
            inputs={bus_connected: solph.Flow()},
            outputs={
                bus_internal: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=block.pois["s2g"].spec_ep_invest,
                        existing=block.sizes["s2g"].preexisting,
                        maximum=block.sizes["s2g"].expansion_max,
                    ),
                    variable_costs=self._cost_eps,
                )
            },
            conversion_factors={bus_internal: 1},
        )
        es.add(block, "inflow", inflow_converter)

        outflow_converter = solph.components.Converter(
            label=_label(block, "outflow"),
            inputs={
                bus_internal: solph.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=block.pois["g2s"].spec_ep_invest,
                        existing=block.sizes["g2s"].preexisting,
                        maximum=block.sizes["g2s"].expansion_max,
                    )
                )
            },
            outputs={bus_connected: solph.Flow()},
            conversion_factors={bus_connected: 1},
        )

        es.add(block, "outflow", outflow_converter)
        es._constraints.add_invest_to_limitation(
            flow=(inflow_converter, bus_internal),
            capex_spec=block.pois["s2g"].capex.spec,
        )
        es._constraints.add_invest_to_limitation(
            flow=(bus_internal, outflow_converter),
            capex_spec=block.pois["g2s"].capex.spec,
        )

        if block.expansion_equal:
            es._constraints.add_equal_invests(
                key=(block.name, "invest"),
                invests=[
                    (inflow_converter, bus_internal),
                    (bus_internal, outflow_converter),
                ],
            )

        for subblock in block.subblocks.values():
            if not isinstance(subblock, blocks.GridMarket):
                raise ValueError(
                    f"Expected each grid connection subblock to be of type {type(blocks.GridMarket)} but got {type(subblock)}"
                )

            self.visit_block(subblock, es, bus_connected=bus_internal)

        # Limit the sum of the power flows of different GridMarkets to the current power of the GridConnection.
        # This ensures that all power being bought or sold has to reach the local energy system and avoids unlimited
        # trading with energy on the different markets without any power limitations.
        # As this model focuses on modeling a local energy system, trading without any physical power flow is not allowed.
        es._constraints.add_equal_flows(
            key=(block.name, "s2g"),
            flow1=(inflow_converter, bus_internal),
        )

        es._constraints.add_equal_flows(
            key=(block.name, "g2s"),
            flow1=(bus_internal, outflow_converter),
        )

        if not block.peakshaving:
            return

        # peak shaving
        # ToDo: limit possible timesteps: enforce period_measurement % scenario.timestep == 0 if period_measurement < scenario.timestep
        peak_bus_in = solph.Bus(label=_label(block, "peak_bus_in"))
        es.add(block, "peak_bus_in", peak_bus_in)

        peak_bus_out = solph.Bus(label=_label(block, "peak_bus_out"))
        es.add(block, "peak_bus_out", peak_bus_out)

        peak_src = solph.components.Source(outputs={peak_bus_in: solph.Flow()})
        es.add(block, "peak_src", peak_src)

        peak_snk = solph.components.Sink(inputs={peak_bus_out: solph.Flow()})
        es.add(block, "peak_snk", peak_snk)

        def create_storage(period):
            activation = block.peak_periods_activation.loc[self._horizon.dti_extd, period.label]
            flush = block.peak_periods_storage_flush.loc[self._horizon.dti]
            soc_limit = block.peak_periods_soc_limit.loc[self._horizon.dti_extd]
            return solph.components.GenericStorage(
                inputs={peak_bus_in: solph.Flow()},
                outputs={
                    peak_bus_out: solph.Flow(
                        maximum=flush * activation,
                        nominal_capacity=solph.Investment(),
                    )
                },
                nominal_capacity=solph.Investment(
                    ep_costs=block.pois[period.label].spec_ep_peak / block.peak_period_measurement.hours,
                    existing=period.peak_power,
                ),
                # use max c-rate to force storage sizing also for measurement duration <= simulation timestep
                invest_relation_output_capacity=1 / self._horizon.timestep.hours,  # empty in single timestep
                initial_storage_level=0.0,
                max_storage_level=(soc_limit * activation),
                balanced=False,
            )

        block.peak_storages = {
            period.label: create_storage(period=period)
            for period in block.peak_periods[
                # only consider intervals which are used in the horizon's simulation period
                (block.peak_periods["start"] < self._horizon.end) & (block.peak_periods["end"] > self._horizon.start)
            ].itertuples(index=False)
        }
        for peak_storage_label, peak_storage in block.peak_storages.items():
            es.add(block, peak_storage_label, peak_storage)

        es._constraints.add_equal_flows(
            key=(block.name, "peakshaving"),
            flow1=(bus_internal, outflow_converter),
            flow2=(peak_src, peak_bus_in),
        )

    def visit_grid_market(
        self, block: blocks.GridMarket, es: OemofEnergySystemContext, bus_connected: solph.Bus
    ) -> None:
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
                    maximum=1 if block.pwr_g2s else None,
                    variable_costs=block.pois["g2s"].spec_ep_operation[self._horizon.dti],
                )
            },
        )
        es.add(block, "src", src)

        snk = solph.components.Sink(
            label=_label(block, "snk"),
            inputs={
                bus_connected: solph.Flow(
                    nominal_capacity=block.pwr_s2g,
                    maximum=1 if block.pwr_s2g else None,
                    variable_costs=block.pois["s2g"].spec_ep_operation[self._horizon.dti],
                )
            },
        )
        es.add(block, "snk", snk)

        # Disallow virtual arbitrage. GridMarket's power flows have to flow through the local energy system.
        es._constraints.add_equal_flows(
            key=(block.parent.name, "s2g"),
            flow2=(bus_connected, snk),
        )

        es._constraints.add_equal_flows(
            key=(block.parent.name, "g2s"),
            flow2=(src, bus_connected),
        )

    def visit_stationary_battery(
        self, block: blocks.StationaryBattery, es: OemofEnergySystemContext, bus_connected: solph.Bus
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
            "storage_balanced": block.balanced if self._scenario.strategy == "go" else False,
            # the stationary battery is the block whose charge timing is otherwise
            # undetermined whenever surplus generation would just be curtailed
            "storage_content_incentive": True,
        }
        self._visit_storage_block(block, es, bus_connected, params)

    def visit_fleet(self, block: blocks.Fleet, es: OemofEnergySystemContext, bus_connected: solph.Bus) -> None:
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

        bus_internal = solph.Bus(label=_label(block, "bus"))
        es.add(block, "bus", bus_internal)

        inflow_converter = solph.components.Converter(
            label=_label(block, "inflow"),
            inputs={
                bus_connected: solph.Flow(
                    variable_costs=block.pois["s2f"].spec_ep_operation[self._horizon.dti],
                    nominal_capacity=block.pwr_lim_s2f,
                    # default value for max is 1; not explicitly set to ensure compatibility with nominal_capacity=None
                )
            },
            outputs={bus_internal: solph.Flow()},
            conversion_factors={bus_internal: 1},
        )
        es.add(block, "inflow", inflow_converter)

        outflow_converter = solph.components.Converter(
            label=_label(block, "outflow"),
            inputs={
                bus_internal: solph.Flow(
                    variable_costs=block.pois["f2s"].spec_ep_operation[self._horizon.dti],
                    nominal_capacity=block.pwr_lim_f2s,
                    # default value for max is 1; not explicitly set to ensure compatibility with nominal_capacity=None
                )
            },
            outputs={bus_connected: solph.Flow(variable_costs=self._cost_eps)},
            conversion_factors={block.bus_connected: 1},
        )
        es.add(block, "outflow", outflow_converter)

        for subblock in block.subblocks.values():
            if not isinstance(subblock, blocks.SubFleet):
                raise ValueError(
                    f"Expected each fleet subblock to be of type {type(blocks.SubFleet)} but got {type(subblock)}"
                )

            self.visit_block(subblock, es, bus_connected=bus_internal)

    def visit_sub_fleet(self, block: blocks.SubFleet, es: OemofEnergySystemContext, bus_connected: solph.Bus) -> None:
        for subblock in block.subblocks.values():
            if isinstance(subblock, blocks.ElectricFleetUnit):
                self.visit_block(subblock, es, bus_connected)

    def visit_electric_fleet_unit(
        self, block: blocks.ElectricFleetUnit, es: OemofEnergySystemContext, bus_connected: solph.Bus
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
        # region calc minimum soc targets before usage and max soc for myopic optimization
        dsoc_ph = block.log.loc[self._horizon.dti, "dsoc"]
        # ensure long tours (> prediction horizon) have enough SOC to fulfil it
        if (block.scenario.strategy == "rh") and (block.mode_scheduling == "oc"):
            soc_min_hor = dsoc_ph.mask(cond=dsoc_ph > 0, other=dsoc_ph + block.dsoc_buffer).clip(
                lower=block.states.loc[self._horizon.dti_extd, "soc_min"],
                upper=block.states.loc[self._horizon.dti_extd, "soc_max"],
            )
        else:  # a priori or global optimization
            soc_min_hor = block.states.loc[self._horizon.dti_extd, "soc_min"]
        block.states.update({"soc_min": soc_min_hor.astype("float64")})
        # endregion

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
            # off for fleet units: rewarding a high SOC here would bias against
            # discharging, i.e. against the V2G the bidirectional modes exist to study
            "storage_content_incentive": False,
        }
        bus_internal = self._visit_storage_block(block, es, bus_connected, params)

        snk = solph.components.Sink(
            label=_label(block, "snk"),
            inputs={bus_internal: solph.Flow(nominal_capacity=1, fix=block.log.loc[self._horizon.dti, "consumption"])},
        )
        es.add(block, "snk", snk)

        bus_ext_ac = solph.Bus(label=_label(block, "bus_ext_ac"))
        es.add(block, "bus_ext_ac", bus_ext_ac)

        src_ext_ac = solph.components.Source(
            label=_label(block, "src_ext_ac"),
            outputs={
                bus_ext_ac: solph.Flow(
                    nominal_capacity=block.pwr_ext_ac_max,
                    maximum=None if block.apriori else block.log.loc[self._horizon.dti, "atac"].astype(int),
                    fix=block.flows_apriori.loc[self._horizon.dti, "p_ext_ac_chg"] if block.apriori else None,
                    variable_costs=block.pois["ext_ac"].spec_ep_operation[self._horizon.dti],
                )
            },
        )
        es.add(block, "src_ext_ac", src_ext_ac)

        conv_ext_ac = solph.components.Converter(
            label=_label(block, "conv_ext_ac"),
            inputs={bus_ext_ac: solph.Flow()},
            outputs={bus_internal: solph.Flow()},
            conversion_factors={bus_internal: block.eff["chg_ac"]},
        )
        es.add(block, "conv_ext_ac", conv_ext_ac)

        bus_ext_dc = solph.Bus(label=_label(block, "bus_ext_dc"))
        es.add(block, "bus_ext_dc", bus_ext_dc)

        src_ext_dc = solph.components.Source(
            label=_label(block, "src_ext_dc"),
            outputs={
                bus_ext_dc: solph.Flow(
                    nominal_capacity=block.pwr_ext_dc_max,
                    maximum=None if block.apriori else block.log.loc[self._horizon.dti, "atdc"].astype(int),
                    fix=block.flows_apriori.loc[self._horizon.dti, "p_ext_dc_chg"] if block.apriori else None,
                    variable_costs=block.pois["ext_dc"].spec_ep_operation[self._horizon.dti],
                )
            },
        )
        es.add(block, "src_ext_dc", src_ext_dc)

        conv_ext_dc = solph.components.Converter(
            label=_label(block, "conv_ext_dc"),
            inputs={bus_ext_dc: solph.Flow()},
            outputs={bus_internal: solph.Flow()},
            conversion_factors={bus_internal: 1},  # billed energy is already dc in external dc charging
        )
        es.add(block, "conv_ext_dc", conv_ext_dc)

        # TODO: Since the constraint is applied for the EFU we need to somehow access
        # the nodes of the storage block.
        # Currently, this works with the labels, but maybe this can be solved cleaner.
        storage = es.get_component(block, "storage")
        inflow_converter = es.get_component(block, "inflow")
        # Ensure that charged energy always flows into the storage and not directly to the ElectricFleetUnit's sink
        # This may happen for on-route charging (simultaneous charging and driving)
        es._constraints.add_equal_flows(
            key=(block.name, "charging"),
            flow1=(bus_internal, storage),
            flows2=[
                (inflow_converter, bus_internal),
                (conv_ext_ac, bus_internal),
                (conv_ext_dc, bus_internal),
            ],
        )

    def _visit_storage_block(
        self,
        block: blocks.StorageBlock,
        es: OemofEnergySystemContext,
        bus_connected: solph.Bus,
        params: dict[str, Any],
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
        es.add(block, "bus", bus_internal)

        inflow = solph.components.Converter(
            label=_label(block, "inflow"),
            inputs={
                bus_connected: solph.Flow(
                    nominal_capacity=params["inflow_nominal_capacity"],
                    maximum=params["inflow_max"],
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
        es.add(block, "inflow", inflow)

        outflow = solph.components.Converter(
            label=_label(block, "outflow"),
            inputs={bus_internal: solph.Flow()},
            outputs={
                bus_connected: solph.Flow(
                    nominal_capacity=params["outflow_nominal_capacity"],
                    maximum=params["outflow_max"],
                    fix=params["outflow_fix"],
                    variable_costs=self._cost_eps
                    * _WASTE_LOOP_PENALTY_MULTIPLIER,  # disincentivize waste loop with inflow (sum must be positive)
                )
            },
            conversion_factors={bus_connected: block.eff["dis_int"]},
        )
        es.add(block, "outflow", outflow)

        storage = solph.components.GenericStorage(
            label=_label(block, "storage"),
            inputs={
                bus_internal: solph.Flow(
                    nominal_capacity=solph.Investment(),
                    variable_costs=block.pois["in"].spec_ep_operation[self._horizon.dti],
                ),
            },
            outputs={
                bus_internal: solph.Flow(nominal_capacity=solph.Investment(), variable_costs=self._cost_eps),
            },
            loss_rate=block.loss_rate_per_hour,
            balanced=params["storage_balanced"],
            # Negative, i.e. a reward for holding energy. A cost on a flow is the same
            # per Wh whenever it is paid, so it cannot express *when* to charge; only a
            # cost on the content, which accrues per timestep the energy sits there, can.
            # Under rolling horizon nothing values energy left at the end of a horizon
            # either, so this doubles as a crude terminal value against end of horizon
            # dumping. Off (0.0) unless the block opts in, see storage_content_incentive.
            storage_costs=(-self._storage_reward_eps if params.get("storage_content_incentive") else 0.0),
            initial_storage_level=block.states.loc[self._horizon.start, ["soc", "soc_min", "soc_max"]].median(),
            # crate measured "outside" of conversion factor (efficiency)
            invest_relation_input_capacity=params["invest_relation_input_capacity"],
            invest_relation_output_capacity=params["invest_relation_output_capacity"],
            inflow_conversion_factor=np.sqrt(block.eff["storage_roundtrip"]),
            outflow_conversion_factor=np.sqrt(block.eff["storage_roundtrip"]),
            nominal_capacity=solph.Investment(
                ep_costs=block.pois["storage"].spec_ep_invest,
                existing=block.sizes["storage"].preexisting,
                maximum=block.sizes["storage"].expansion_max,
            ),
            max_storage_level=block.states.loc[self._horizon.dti_extd, "soc_max"],
            min_storage_level=block.states.loc[self._horizon.dti_extd, "soc_min"],
        )
        es.add(block, "storage", storage)

        es._constraints.add_invest_to_limitation(
            storage=storage,
            capex_spec=block.pois["storage"].capex.spec,
        )

        return bus_internal


def _label(block: blocks.BaseBlock, label: str) -> str:
    return f"{block.name}-{label}"
