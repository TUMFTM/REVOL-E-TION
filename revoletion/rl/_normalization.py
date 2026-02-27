from __future__ import annotations

import typing

import numpy as np
import pandas as pd

from revoletion import blocks

from . import _context as context

_NormalizationConstants = dict[blocks.BaseBlock, float]

_V = typing.TypeVar("_V", np.ndarray, float, pd.Series)

_QUANTILE = 0.95


class NormalizationProvider:
    def __init__(
        self,
        normalization_constants_input_opex: _NormalizationConstants | None = None,
        normalization_constants_input_power: _NormalizationConstants | None = None,
        normalization_constants_output_opex: _NormalizationConstants | None = None,
        normalization_constants_output_power: _NormalizationConstants | None = None,
        normalization_constants_energy: _NormalizationConstants | None = None,
    ) -> None:
        self._normalization_constants_input_opex = normalization_constants_input_opex or {}
        self._normalization_constants_input_power = normalization_constants_input_power or {}
        self._normalization_constants_output_opex = normalization_constants_output_opex or {}
        self._normalization_constants_output_power = normalization_constants_output_power or {}
        self._normalization_constants_energy = normalization_constants_energy or {}

    @classmethod
    def from_ctx(cls, ctx: context.Context) -> NormalizationProvider:
        normalization_constants_input_opex = {}
        normalization_constants_input_power = {}
        normalization_constants_output_opex = {}
        normalization_constants_output_power = {}
        normalization_constants_energy = {}

        for controllable_source_block in ctx.controllable_source_blocks:
            normalization_constants_output_power[controllable_source_block] = controllable_source_block.sizes[
                "block"
            ].preexisting
            max_opex_per_unit = controllable_source_block.evaluators["block"].opt.spec_ep_invest.quantile(_QUANTILE)
            max_opex = controllable_source_block.sizes["block"].preexisting * max_opex_per_unit
            normalization_constants_output_opex[controllable_source_block] = abs(max_opex)

        for grid_market_block in ctx.grid_market_blocks:
            normalization_constants_input_power[grid_market_block] = abs(grid_market_block.pwr_g2s)

            max_import_costs_per_unit = grid_market_block.evaluators["g2s"].opt.spec_ep_operation.quantile(_QUANTILE)
            max_import_costs = grid_market_block.pwr_g2s * max_import_costs_per_unit
            normalization_constants_input_opex[grid_market_block] = abs(max_import_costs)

            normalization_constants_output_power[grid_market_block] = abs(grid_market_block.pwr_s2g)

            max_export_profit_per_unit = grid_market_block.evaluators["s2g"].opt.spec_ep_operation.quantile(
                1.0 - _QUANTILE
            )
            max_export_profit = grid_market_block.pwr_s2g * max_export_profit_per_unit
            normalization_constants_output_opex[grid_market_block] = abs(max_export_profit)

        for renewable_source_block in ctx.renewable_source_blocks:
            max_pwr = renewable_source_block.sizes["block"].preexisting
            normalization_constants_output_power[renewable_source_block] = max_pwr
            max_cost_per_unit = renewable_source_block.evaluators["block"].opt.spec_ep_operation.quantile(_QUANTILE)
            normalization_constants_output_opex[renewable_source_block] = abs(max_cost_per_unit * max_pwr)

        for fixed_demand_block in ctx.fixed_demand_blocks:
            max_fixed_demand = fixed_demand_block.flows_apriori["demand"].quantile(_QUANTILE)
            normalization_constants_input_power[fixed_demand_block] = abs(max_fixed_demand)

        for stationary_battery_block in ctx.stationary_battery_blocks:
            max_energy_wh = stationary_battery_block.sizes["storage"].preexisting
            normalization_constants_energy[stationary_battery_block] = max_energy_wh

            max_power = max_energy_wh * ctx.time.horizon.timestep.hours
            normalization_constants_input_power[stationary_battery_block] = max_power
            normalization_constants_output_power[stationary_battery_block] = max_power

        return cls(
            normalization_constants_input_opex,
            normalization_constants_input_power,
            normalization_constants_output_opex,
            normalization_constants_output_power,
            normalization_constants_energy,
        )

    def normalize_input_power(self, orig_power: _V, block: blocks.BaseBlock) -> _V:
        if block not in self._normalization_constants_input_power:
            raise ValueError(
                f"Failed to normalize input power for block {block.name}: No normalization constant registered for the block"
            )
        normalization_constant = self._normalization_constants_input_power[block]
        return np.clip(orig_power / normalization_constant, -1.0, 1.0)

    def normalize_output_power(self, orig_power: _V, block: blocks.BaseBlock) -> _V:
        if block not in self._normalization_constants_output_power:
            raise ValueError(
                f"Failed to normalize output power for block {block.name}: No normalization constant registered for the block"
            )
        normalization_constant = self._normalization_constants_output_power[block]
        return np.clip(orig_power / normalization_constant, -1.0, 1.0)

    def normalize_input_opex(self, orig_opex: _V, block: blocks.BaseBlock) -> _V:
        if block not in self._normalization_constants_input_opex:
            raise ValueError(
                f"Failed to normalize input opex for block '{block.name}': No normalization constant registered for the block"
            )
        normalization_constant = self._normalization_constants_input_opex[block]
        if normalization_constant == 0.0:
            return orig_opex
        return orig_opex / normalization_constant

    def normalize_output_opex(self, orig_opex: _V, block: blocks.BaseBlock) -> _V:
        if block not in self._normalization_constants_output_opex:
            raise ValueError(
                f"Failed to normalize output opex for block {block.name}: No normalization constant registered for the block"
            )
        normalization_constant = self._normalization_constants_output_opex[block]
        if normalization_constant == 0.0:
            return orig_opex
        return orig_opex / normalization_constant

    def normalize_stored_energy(self, stored_energy: _V, block: blocks.BaseBlock) -> _V:
        if block not in self._normalization_constants_energy:
            raise ValueError(
                f"Failed to normalize stored energy for block {block.name}: No normalization constant registered for the block"
            )
        normalization_constant = self._normalization_constants_energy[block]
        return np.clip(stored_energy / normalization_constant, 0.0, 1.0)
