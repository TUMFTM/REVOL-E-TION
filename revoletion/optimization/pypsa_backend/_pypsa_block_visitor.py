"""Module which handles the construction of a PyPSA network from a REVOL-E-TION block structure."""

import logging

import numpy as np
import pandas as pd
import pypsa
from typing_extensions import override

from revoletion import blocks, utils

from ._pypsa_net_builder import PyPSANetworkBuilder

_LOGGER = logging.getLogger(__name__)


class PyPSABlockVisitor(blocks.BlockVisitor[None]):
    """
    Visitor to construct a PyPSA Network from a REVOL-E-TION block structure.
    """

    # Since each top level block connects to either the AC or DC bus but they are not children
    # of the `SystemCore` block, the names for the core buses are fixed.
    # This way, each block can be easily assigned to the AC or DC core bus.
    _CORE_AC_BUS_NAME = "core-ac-bus"
    _CORE_DC_BUS_NAME = "core-dc-bus"

    _AC_CARRIER = "AC"
    _DC_CARRIER = "DC"

    def __init__(
        self,
        horizon: utils.TimeSettings,
        cost_eps: float,
        enable_investment: bool = True,
        enable_fixed_dispatch: bool = True,
        enforce_soc_min: bool = True,
    ):
        """
        :param datetime_index: The datetime index covering the scenario data. Required to correctly initialize the time series data in the PyPSA network.
        :param cost_eps:
        :param external_charging: Whether external charging should be enabled for EVs.
        """
        self._horizon = horizon
        self._datetime_index = horizon.dti
        self._cost_eps = cost_eps
        self._enable_investment = enable_investment
        self._enable_fixed_dispatch = enable_fixed_dispatch
        self._enforce_soc_min = enforce_soc_min

    def create_pypsa_network(self, block_registry: dict[str, dict[str, blocks.BaseBlock]]) -> pypsa.Network:
        builder = PyPSANetworkBuilder(self._horizon)
        # Register the carriers with PyPSA. While this does not change anything for AC and DC
        # it makes PyPSA stop complaining.
        builder.add_carrier(self._AC_CARRIER)
        builder.add_carrier(self._DC_CARRIER)

        for block in block_registry.get("TopLevelBlock", {}).values():
            self.visit_block(block, builder=builder)
        return builder.build()

    @override
    def visit_block(self, block: blocks.BaseBlock, *args, **kwargs) -> None:
        # All top-level blocks except the system core must be connected to the system cores AC or DC bus.
        # This injects the correct bus into the parameters for each block construction method.
        if hasattr(block, "system"):
            bus_connected_label = self._CORE_AC_BUS_NAME if block.system == "ac" else self._CORE_DC_BUS_NAME
            kwargs["bus_connected"] = bus_connected_label
            kwargs["bus_carrier"] = self._AC_CARRIER if block.system == "ac" else self._DC_CARRIER

        match block:
            case blocks.SystemCore():
                self.visit_system_core(block, *args, **kwargs)
            case blocks.GridConnection():
                self.visit_grid_connection(block, *args, **kwargs)
            case blocks.RenewableSource():
                self.visit_renewable_source(block, *args, **kwargs)
            case blocks.ControllableSource():
                self.visit_controllable_source(block, *args, **kwargs)
            case blocks.FixedDemand():
                self.visit_fixed_demand(block, *args, **kwargs)
            case blocks.StationaryBattery():
                self.visit_stationary_battery(block, *args, **kwargs)
            case blocks.Fleet():
                self.visit_fleet(block, *args, **kwargs)
            case _:
                raise RuntimeError(f"Cannot visit block {block.name}: block of type {type(block)} is not supported")

    def visit_system_core(self, block: blocks.SystemCore, builder: PyPSANetworkBuilder) -> None:
        # Define the core buses for AC and DC. All other blocks most directly or indirectly connect to one of those buses.
        builder.add_bus(name=self._CORE_AC_BUS_NAME, carrier=self._AC_CARRIER)
        builder.add_bus(name=self._CORE_DC_BUS_NAME, carrier=self._DC_CARRIER)

        # The AC/DC converter is modeled as a lossy link between the AC and DC bus.
        builder.add_link(
            name=make_pypsa_label(block, "acdc-link"),
            bus0=self._CORE_AC_BUS_NAME,
            bus1=self._CORE_DC_BUS_NAME,
            # p_nom is set for use cases when investment is disabled and is ignored if investment is enabled.
            # For investment cases p_nom_min is used instead of p_nom.
            p_nom=block.sizes["acdc"].preexisting,
            p_nom_min=block.sizes["acdc"].preexisting,
            # The `Size` object converts a `None` maximum expansion to 0. If this is passed to PyPSA, it might break the optimization.
            p_nom_max=block.sizes["acdc"].expansion_max if block.sizes["acdc"].invest else None,
            p_nom_extendable=self._enable_investment and block.sizes["acdc"].invest,
            efficiency=block.eff["acdc"],
            capital_cost=block.evaluators["acdc"].opt.spec_ep_invest,
            marginal_cost=block.evaluators["acdc"].opt.spec_ep_operation[self._datetime_index] + self._cost_eps,
        )

        # The DC/AC converter is modeled as a lossy link between the DC and AC bus.
        builder.add_link(
            name=make_pypsa_label(block, "dcac-link"),
            bus0=self._CORE_DC_BUS_NAME,
            bus1=self._CORE_AC_BUS_NAME,
            # p_nom is set for use cases when investment is disabled and is ignored if investment is enabled.
            # For investment cases p_nom_min is used instead of p_nom.
            p_nom=block.sizes["dcac"].preexisting,
            p_nom_min=block.sizes["dcac"].preexisting,
            # The `Size` object converts a `None` maximum expansion to 0. If this is passed to PyPSA, it might break the optimization.
            p_nom_max=block.sizes["dcac"].expansion_max if block.sizes["dcac"].invest else None,
            p_nom_extendable=self._enable_investment and block.sizes["dcac"].invest,
            efficiency=block.eff["dcac"],
            capital_cost=block.evaluators["dcac"].opt.spec_ep_invest,
            marginal_cost=block.evaluators["dcac"].opt.spec_ep_operation[self._datetime_index] + self._cost_eps,
        )

    def visit_grid_connection(
        self, block: blocks.GridConnection, builder: PyPSANetworkBuilder, bus_connected: str, bus_carrier: str
    ) -> None:
        bus_grid = make_pypsa_label(block, "grid-bus")
        builder.add_bus(name=bus_grid, carrier=bus_carrier)

        # Create link for energy from the site to the grid.
        builder.add_link(
            name=make_pypsa_label(block, "inflow-link"),
            bus0=bus_connected,
            bus1=bus_grid,
            # p_nom is set for use cases when investment is disabled and is ignored if investment is enabled.
            # For investment cases p_nom_min is used instead of p_nom.
            p_nom=block.sizes["s2g"].preexisting,
            p_nom_min=block.sizes["s2g"].preexisting,
            # The `Size` object converts a `None` maximum expansion to 0. If this is passed to PyPSA, it might break the optimization.
            p_nom_max=block.sizes["s2g"].expansion_max if block.sizes["s2g"].invest else None,
            p_nom_extendable=self._enable_investment and block.sizes["s2g"].invest,
            marginal_cost=self._cost_eps,
            capital_cost=block.evaluators["s2g"].opt.spec_ep_invest,
        )

        # TODO: add peakshaving.

        # Link for energy from the site to the grid.
        builder.add_link(
            name=make_pypsa_label(block, "outflow-link"),
            bus0=bus_grid,
            bus1=bus_connected,
            # p_nom is set for use cases when investment is disabled and is ignored if investment is enabled.
            # For investment cases p_nom_min is used instead of p_nom.
            p_nom=block.sizes["g2s"].preexisting,
            p_nom_min=block.sizes["g2s"].preexisting,
            # The `Size` object converts a `None` maximum expansion to 0. If this is passed to PyPSA, it might break the optimization.
            p_nom_max=block.sizes["g2s"].expansion_max if block.sizes["g2s"].invest else None,
            p_nom_extendable=self._enable_investment and block.sizes["g2s"].invest,
            marginal_cost=self._cost_eps,
            capital_cost=block.evaluators["g2s"].opt.spec_ep_invest,
        )

        for subblock in block.subblocks.values():
            if not isinstance(subblock, blocks.GridMarket):
                raise RuntimeError(
                    f"Expected subblock {subblock.name} of block {block.name} to be of type {blocks.GridMarket}, but got {type(subblock)}"
                )
            self.visit_grid_market(subblock, builder, bus_grid)

    def visit_grid_market(self, block: blocks.GridMarket, builder: PyPSANetworkBuilder, bus_connected: str) -> None:
        variable_cost_g2s = block.evaluators["g2s"].opt.spec_ep_operation[self._datetime_index]
        builder.add_generator(
            name=make_pypsa_label(block, "import-gen"),
            bus=bus_connected,
            marginal_cost=variable_cost_g2s,
            # How much energy can the system buy from this market?
            p_nom=block.pwr_g2s if block.pwr_g2s else np.inf,
            # The generator is added unconditionally to the network.
            # This controls, whether the generator can actually buy energy from this market.
            p_max_pu=1.0 if block.pwr_g2s else np.nan,
            # Mark this as the Slack, since PyPSA should just pull (buy) any missing energy here.
            control="Slack",
            # Only the links of the parent `GridConnection` are extendable.
            p_nom_extendable=False,
        )

        # Generator with negative sign to absorb (sell) energy.
        variable_cost_s2g = block.evaluators["s2g"].opt.spec_ep_operation[self._datetime_index]
        builder.add_generator(
            name=make_pypsa_label(block, "export-gen"),
            bus=bus_connected,
            marginal_cost=variable_cost_s2g,
            # How much energy can the system sell to this market?
            p_nom=block.pwr_s2g if block.pwr_s2g else np.inf,
            # The generator is added unconditionally to the network.
            # This controls, whether the generator can actually sell energy to this market.
            p_max_pu=1.0 if block.pwr_s2g else np.nan,
            # Mark this as the Slack, since PyPSA should just dump (sell) any excess energy here.
            control="Slack",
            # Only the links of the parent `GridConnection` are extendable.
            p_nom_extendable=False,
            # Consuming generator must have negative sign.
            sign=-1,
        )

    def visit_renewable_source(
        self, block: blocks.RenewableSource, builder: PyPSANetworkBuilder, bus_connected: str, bus_carrier: str
    ) -> None:
        bus_internal = make_pypsa_label(block, "internal-bus")
        builder.add_bus(name=bus_internal, carrier=bus_carrier)

        builder.add_link(
            name=make_pypsa_label(block, "outflow-link"),
            bus0=bus_internal,
            bus1=bus_connected,
            efficiency=block.eff["block"],
            # The link should be able to carry all power generated by the renewable source, even after expansion.
            p_nom=np.inf,
            p_nom_extendable=False,
        )

        power_spec = block.data.loc[self._datetime_index, "power_spec"]
        variable_costs = block.evaluators["block"].opt.spec_ep_operation[self._datetime_index]

        # oemof models curtailment (`curt`) for renewable sources using an excess load, where excess energy is send to.
        # For PyPSA we instead define a p_max_pu, indicating a maximum (`pot`) power and let PyPSA
        # decide how much it wants to use.
        # Later on, we can determine the curtailment from the `pot` power and the `out` power.
        builder.add_generator(
            name=make_pypsa_label(block, "gen"),
            bus=bus_internal,
            p_max_pu=power_spec,
            # p_nom is set for use cases when investment is disabled and is ignored if investment is enabled.
            # For investment cases p_nom_min is used instead of p_nom.
            p_nom=block.sizes["block"].preexisting,
            p_nom_min=block.sizes["block"].preexisting,
            # The `Size` object converts a `None` maximum expansion to 0. If this is passed to PyPSA, it might break the optimization.
            p_nom_max=block.sizes["block"].expansion_max if block.sizes["block"].invest else None,
            p_nom_extendable=self._enable_investment and block.sizes["block"].invest,
            marginal_cost=variable_costs,
            control="PQ",
            capital_cost=block.evaluators["block"].opt.spec_ep_invest,
        )

    def visit_controllable_source(
        self, block: blocks.ControllableSource, builder: PyPSANetworkBuilder, bus_connected: str, bus_carrier: str
    ) -> None:
        variable_costs = block.evaluators["block"].opt.spec_ep_operation[self._datetime_index]
        builder.add_generator(
            name=make_pypsa_label(block, "gen"),
            bus=bus_connected,
            # p_nom is set for use cases when investment is disabled and is ignored if investment is enabled.
            # For investment cases p_nom_min is used instead of p_nom.
            p_nom=block.sizes["block"].preexisting,
            p_nom_min=block.sizes["block"].preexisting,
            # The `Size` object converts a `None` maximum expansion to 0. If this is passed to PyPSA, it might break the optimization.
            p_nom_max=block.sizes["block"].expansion_max if block.sizes["block"].invest else None,
            p_nom_extendable=self._enable_investment and block.sizes["block"].invest,
            marginal_cost=variable_costs,
            control="PQ",
            capital_cost=block.evaluators["block"].opt.spec_ep_invest,
        )

    def visit_fixed_demand(
        self, block: blocks.FixedDemand, builder: PyPSANetworkBuilder, bus_connected: str, bus_carrier: str
    ) -> None:
        bus_load_name = make_pypsa_label(block, "load-bus")
        builder.add_bus(name=bus_load_name, carrier=bus_carrier)

        builder.add_link(
            name=make_pypsa_label(block, "inflow-link"),
            bus0=bus_connected,
            bus1=bus_load_name,
            p_nom=np.inf,
        )

        demand_fix = block.flows_apriori["demand"][self._datetime_index]
        builder.add_load(
            name=make_pypsa_label(block, "load"),
            bus=bus_load_name,
            p_set=demand_fix,
        )

    def visit_stationary_battery(
        self, block: blocks.StationaryBattery, builder: PyPSANetworkBuilder, bus_connected: str, bus_carrier: str
    ) -> None:
        bus_battery = make_pypsa_label(block, "battery-bus")
        builder.add_bus(name=bus_battery, carrier=bus_carrier)

        # For OMEOF the inflow variable costs are defined at two locations:
        # once for the inflow converter and once for the input of the storage itself.
        # For PyPSA we do not have separate outputs/inputs for a store, so the costs are just summed and applied only to the inflow link.
        inflow_variable_costs = self._cost_eps * -3 + block.evaluators["in"].opt.spec_ep_operation[self._datetime_index]
        builder.add_link(
            name=make_pypsa_label(block, "inflow-link"),
            bus0=bus_connected,
            bus1=bus_battery,
            # For OMEOF the capacity of the links is set to None.
            p_nom=np.inf,
            marginal_cost=inflow_variable_costs,
            efficiency=block.eff["chg_int"],
        )

        # For OMEOF the outflow variable costs are defined at two locations:
        # once for the outflow converter and once for the output of the storage itself.
        # For PyPSA we do not have separate outputs/inputs for a store, so the costs are just summed and applied only to the outflow link.
        outflow_variable_costs = self._cost_eps * 4 + self._cost_eps
        builder.add_link(
            name=make_pypsa_label(block, "outflow-link"),
            bus0=bus_battery,
            bus1=bus_connected,
            # For OMEOF the capacity of the links is set to None.
            p_nom=np.inf,
            marginal_cost=outflow_variable_costs,
            efficiency=block.eff["dis_int"],
        )

        soc_initial_percent = block.states.loc[self._datetime_index, "soc"].median()
        battery_capacity_w = block.sizes["storage"].preexisting
        battery_energy_initial_w = battery_capacity_w * soc_initial_percent
        builder.add_store(
            name=make_pypsa_label(block, "battery-store"),
            bus=bus_battery,
            standing_loss=block.loss_rate_per_hour,
            # e_nom is set for use cases when investment is disabled and is ignored if investment is enabled.
            # For investment cases e_nom_min is used instead of p_nom.
            e_nom=battery_capacity_w,
            e_nom_min=battery_capacity_w,
            # The `Size` object converts a `None` maximum expansion to 0. If this is passed to PyPSA, it might break the optimization.
            e_nom_max=block.sizes["storage"].expansion_max if block.sizes["storage"].invest else None,
            e_initial=battery_energy_initial_w,
            e_nom_extendable=self._enable_investment and block.sizes["storage"].invest,
            capital_cost=block.evaluators["storage"].opt.spec_ep_invest,
        )

    def visit_fleet(
        self, block: blocks.Fleet, builder: PyPSANetworkBuilder, bus_connected: str, bus_carrier: str
    ) -> None:
        bus_fleet_name = make_pypsa_label(block, "fleet-bus")
        builder.add_bus(name=bus_fleet_name, carrier=bus_carrier)

        variable_costs = block.evaluators["s2f"].opt.spec_ep_operation[self._datetime_index]
        builder.add_link(
            name=make_pypsa_label(block, "inflow-link"),
            bus0=bus_connected,
            bus1=bus_fleet_name,
            p_nom=block.pwr_lim_s2f,
            p_nom_extendable=False,
            efficiency=1.0,
            marginal_cost=variable_costs,
        )

        variable_costs = block.evaluators["f2s"].opt.spec_ep_operation[self._datetime_index] + self._cost_eps
        builder.add_link(
            name=make_pypsa_label(block, "outflow-link"),
            bus0=bus_fleet_name,
            bus1=bus_connected,
            p_nom=block.pwr_lim_f2s,
            p_nom_extendable=False,
            efficiency=1.0,
            marginal_cost=variable_costs,
        )

        for subblock in block.subblocks.values():
            if not isinstance(subblock, blocks.SubFleet):
                raise ValueError(
                    f"Expected each fleet subblock to be of type {type(blocks.SubFleet)} but got {type(subblock)}"
                )

            self.visit_sub_fleet(subblock, builder, bus_fleet_name)

    def visit_sub_fleet(self, block: blocks.SubFleet, builder: PyPSANetworkBuilder, bus_connected: str) -> None:
        for subblock in block.subblocks.values():
            if isinstance(subblock, blocks.ElectricFleetUnit):
                self.visit_electric_fleet_unit(subblock, builder, bus_connected)

    def visit_electric_fleet_unit(
        self, block: blocks.ElectricFleetUnit, builder: PyPSANetworkBuilder, bus_connected: str
    ) -> None:
        bus_efu_name = make_pypsa_label(block, "efu-bus")
        builder.add_bus(name=bus_efu_name)

        # At-Site Charging

        inflow_capacity = block.pwr_chg_max
        inflow_max = (
            pd.Series(1.0, index=self._datetime_index)
            if block.apriori
            else block.log.loc[self._datetime_index, "atbase"].astype(int)
        )
        inflow_fix = (
            block.flows_apriori.loc[self._datetime_index, "p_int_chg"] * inflow_capacity
            if self._enable_fixed_dispatch and block.apriori
            else pd.Series(np.nan, index=self._datetime_index)
        )
        # The variable costs are fixed similar to how it is defined for OEMOF.
        inflow_variable_costs = self._cost_eps * -3
        builder.add_link(
            name=make_pypsa_label(block, "inflow-link"),
            bus0=bus_connected,
            bus1=bus_efu_name,
            p_nom=inflow_capacity,
            # Ensure p_min_pu is populated, because it might be modified later.
            p_min_pu=pd.Series(0.0, index=self._datetime_index),
            p_max_pu=inflow_max,
            p_set=inflow_fix,
            efficiency=block.eff["chg_int"],
            marginal_cost=inflow_variable_costs,
        )

        outflow_capacity = block.pwr_dis_max * block.eff["dis_int"]
        outflow_max = (
            pd.Series(1.0, index=self._datetime_index)
            if block.apriori
            else block.log.loc[self._datetime_index, "atbase"].astype(int)
        )
        outflow_fix = (
            block.flows_apriori.loc[self._datetime_index, "p_int_dis"] * outflow_capacity
            if self._enable_fixed_dispatch and block.apriori
            else pd.Series(np.nan, index=self._datetime_index)
        )
        # The variable costs are fixed similar to how it is defined for OEMOF.
        outflow_variable_costs = self._cost_eps * 4 + self._cost_eps
        builder.add_link(
            name=make_pypsa_label(block, "outflow-link"),
            bus0=bus_efu_name,
            bus1=bus_connected,
            p_nom=outflow_capacity,
            p_min_pu=pd.Series(0.0, index=self._datetime_index),
            p_max_pu=outflow_max,
            p_set=outflow_fix,
            efficiency=block.eff["dis_int"],
            marginal_cost=outflow_variable_costs,
        )

        # External AC/DC Charging

        bus_ext_ac = make_pypsa_label(block, "ext-ac-bus")
        builder.add_bus(bus_ext_ac)
        ext_ac_capacity = block.pwr_ext_ac_max * block.eff["chg_ac"]
        max_ext_ac = (
            pd.Series(1.0, index=self._datetime_index)
            if block.apriori
            else block.log.loc[self._datetime_index, "atac"].astype(int)
        )
        fix_ext_ac = (
            block.flows_apriori.loc[self._datetime_index, "p_ext_ac_chg"]
            if block.apriori and self._enable_fixed_dispatch
            else pd.Series(np.nan, index=self._datetime_index)
        )
        variable_costs_ext_ac = block.evaluators["ext_ac"].opt.spec_ep_operation[self._datetime_index]
        builder.add_link(
            name=make_pypsa_label(block, "ext-ac-inflow-link"),
            bus0=bus_ext_ac,
            bus1=bus_efu_name,
            efficiency=block.eff["chg_ac"],
            p_nom=ext_ac_capacity,
        )
        builder.add_generator(
            name=make_pypsa_label(block, "ext-ac-gen"),
            bus=bus_ext_ac,
            p_nom=ext_ac_capacity,
            p_set=fix_ext_ac,
            p_max_pu=max_ext_ac,
            marginal_cost=variable_costs_ext_ac,
        )

        bus_ext_dc = make_pypsa_label(block, "ext-dc-bus")
        builder.add_bus(bus_ext_dc)
        ext_dc_capacity = block.pwr_ext_dc_max
        max_ext_dc = (
            pd.Series(1.0, index=self._datetime_index)
            if block.apriori
            else block.log.loc[self._datetime_index, "atdc"].astype(int)
        )
        fix_ext_dc = (
            block.flows_apriori.loc[self._datetime_index, "p_ext_dc_chg"]
            if block.apriori and self._enable_fixed_dispatch
            else pd.Series(np.nan, index=self._datetime_index)
        )
        variable_costs_ext_dc = block.evaluators["ext_dc"].opt.spec_ep_operation[self._datetime_index]
        builder.add_link(
            name=make_pypsa_label(block, "ext-dc-inflow-link"),
            bus0=bus_ext_dc,
            bus1=bus_efu_name,
            # billed energy is already dc in external dc charging
            efficiency=1,
            p_nom=ext_dc_capacity,
        )
        builder.add_generator(
            name=make_pypsa_label(block, "ext-dc-gen"),
            bus=bus_ext_dc,
            p_nom=ext_dc_capacity,
            p_set=fix_ext_dc,
            p_max_pu=max_ext_dc,
            marginal_cost=variable_costs_ext_dc,
        )

        # Battery

        battery_soc_initial_percent = block.states.loc[self._datetime_index[0], "soc"]
        battery_capacity_wh = block.sizes["storage"].preexisting
        battery_e_initial_wh = (
            battery_capacity_wh
            if np.isnan(battery_soc_initial_percent)
            else battery_capacity_wh * battery_soc_initial_percent
        )
        # In oemof the storage has a dedicated inflow and outflow efficiency. By default
        # PyPSA does not support this, so additional links are required.
        bus_battery_name = make_pypsa_label(block, "battery-bus")
        builder.add_bus(bus_battery_name)
        max_int_chg = max(ext_dc_capacity, ext_ac_capacity, inflow_capacity)
        builder.add_link(
            name=make_pypsa_label(block, "battery-int-chg-link"),
            bus0=bus_efu_name,
            bus1=bus_battery_name,
            p_nom=max_int_chg,
            efficiency=np.sqrt(
                block.eff["storage_roundtrip"],
            ),
            # Stop PyPSA from creating circular flows.
            marginal_cost=self._cost_eps,
        )

        max_int_dis = outflow_capacity
        builder.add_link(
            name=make_pypsa_label(block, "battery-int-dis-link"),
            bus0=bus_battery_name,
            bus1=bus_efu_name,
            p_nom=max_int_dis,
            efficiency=np.sqrt(
                block.eff["storage_roundtrip"],
            ),
            # Stop PyPSA from creating circular flows.
            marginal_cost=self._cost_eps,
        )
        builder.add_store(
            name=make_pypsa_label(block, "battery-store"),
            bus=bus_battery_name,
            e_nom=battery_capacity_wh,
            e_nom_min=battery_capacity_wh,
            # The `Size` object converts a `None` maximum expansion to 0. If this is passed to PyPSA, it might break the optimization, since then e_nom_max < e_nom_min.
            e_nom_max=block.sizes["storage"].expansion_max if block.sizes["storage"].invest else None,
            e_nom_extendable=self._enable_investment and block.sizes["storage"].invest,
            e_initial=battery_e_initial_wh,
            e_min_pu=block.states.loc[self._datetime_index, "soc_min"] if self._enforce_soc_min else None,
            e_max_pu=block.states.loc[self._datetime_index, "soc_max"],
            standing_loss=block.loss_rate_per_ts,
            marginal_cost=block.evaluators["in"].opt.spec_ep_operation[self._datetime_index],
            capital_cost=block.evaluators["storage"].opt.spec_ep_invest,
        )

        load_battery_power = block.log.loc[self._datetime_index, "consumption"]
        builder.add_load(
            name=make_pypsa_label(block, "battery-load"),
            bus=bus_efu_name,
            p_set=load_battery_power,
        )


def make_pypsa_label(block: blocks.BaseBlock, label: str) -> str:
    return f"{block.name}-{label}"
