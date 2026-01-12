import gymnasium as gym
import numpy as np
import pandas as pd

from revoletion import optimization

from . import _context as context
from . import _forecast_provider as forecast_provider
from . import utils as rl_utils

# Keys in the observation space for consistent access in custom agents.
OBS_KEY_TIME_FEATURES = "time_of_day"
OBS_KEY_EFUS_AVAILABLE = "efus_available"
OBS_KEY_EFUS_SOC = "efus_soc"
OBS_KEY_EFUS_CURRENT_SOC_DIFF = "efus_soc_diff"
OBS_KEY_EFUS_REQUIRED_SOCS = "efus_required_socs"
OBS_KEY_EFUS_REAL_POWER_UNIT = "efus_real_power_unit"
OBS_KEY_EFUS_NEXT_REQUIRED_SOC_DIFF = "efus_next_required_soc_diff"
OBS_KEY_EFUS_AVAILABLE_NOW = "efus_avalable_now"
OBS_KEY_EFUS_URGENCY = "efus_urgency"

OBS_KEY_FLEETS_IN_POWER = "fleets_in_power"
OBS_KEY_FLEETS_OUT_POWER = "fleets_out_power"

OBS_KEY_RENEWABLES_POWER = "renewables_power"
OBS_KEY_RENEWABLES_SCHEDULE = "renewables_schedule"

OBS_KEY_FIXED_DEMANDS = "demands_schedule"

OBS_KEY_GRID_IMPORT_COSTS = "grid_import_costs"
OBS_KEY_GRID_IMPORT_POWER = "grid_import_power"
OBS_KEY_GRID_EXPORT_COSTS = "grid_export_costs"
OBS_KEY_GRID_EXPORT_POWER = "grid_export_power"

OBS_KEY_STATIONARY_BATTERIES_SOC = "stationary_batteries_soc"

OBS_KEY_CONTROLLABLE_SOURCES_POWER = "controllable_sources_power"


class EnvironmentFeatureExtractor:
    def __init__(self, forecast_provider: forecast_provider.ForecastProvider, soc_min: float) -> None:
        self._forecast_provider = forecast_provider
        self._soc_min = soc_min

    def extract_all_features(
        self,
        ctx: context.Context,
        optimization_result: optimization.OptimizationResult | None = None,
    ) -> dict[str, np.ndarray]:
        state_dict = {}

        state_dict.update(self.get_time_features(ctx))

        state_dict.update(self.get_fleets_features(ctx, optimization_result))

        if ctx.has_grid_connection:
            state_dict.update(self.get_grid_markets_features(ctx, optimization_result))

        if ctx.has_renewable_sources:
            state_dict.update(self.get_renewable_sources_features(ctx, optimization_result))

        if ctx.has_controllable_sources:
            state_dict.update(self.get_controllable_sources_features(ctx, optimization_result))

        if ctx.has_stationary_batteries:
            state_dict.update(self.get_stationary_batteries_features(ctx, optimization_result))

        if ctx.has_fixed_demands:
            state_dict.update(self.get_fixed_demands_features(ctx))

        return state_dict

    def get_time_features(self, ctx: context.Context) -> dict[str, np.ndarray]:
        # Hour of day (0-23)
        hour = ctx.current_time_step.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Day of week (0-6)
        day_of_week = ctx.current_time_step.dayofweek
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)

        day_of_year = ctx.current_time_step.dayofyear
        doy_sin = np.sin(2 * np.pi * day_of_year / 365)
        doy_cos = np.cos(2 * np.pi * day_of_year / 365)

        return {
            OBS_KEY_TIME_FEATURES: np.array([hour_sin, hour_cos, dow_sin, dow_cos, doy_sin, doy_cos], dtype=np.float32)
        }

    def get_fleets_features(
        self,
        ctx: context.Context,
        optimization_result: optimization.OptimizationResult | None = None,
    ) -> dict[str, np.ndarray]:
        fleet_in_power_units = []
        fleet_out_power_units = []

        efus_soc = []
        efus_current_soc_diff = []
        efus_next_required_soc_diff = []
        efus_urgency = []
        efus_available_now = []

        efus_real_power_unit = []
        efus_required_soc = []
        efus_available_forecast = []

        for fleet_block, electric_fleet_unit_blocks in ctx.electric_fleets.items():
            if optimization_result is not None:
                fleet_power_flow = optimization_result.get_power_flow(fleet_block, ctx.previous_time_step)

                fleet_in_power_unit = fleet_power_flow["in"] / fleet_block.pwr_lim_s2f
                fleet_out_power_unit = fleet_power_flow["out"] / fleet_block.pwr_lim_f2s

                fleet_in_power_units.append(fleet_in_power_unit)
                fleet_out_power_units.append(fleet_out_power_unit)
            else:
                fleet_in_power_units.append(0.0)
                fleet_out_power_units.append(0.0)

            for electric_fleet_unit_block in electric_fleet_unit_blocks:
                if optimization_result is None:
                    soc = electric_fleet_unit_block.states.loc[ctx.current_time_step, "soc"]
                    if np.isnan(soc):
                        soc = 0.0

                    real_power_unit = 0.0
                else:
                    stored_energy = optimization_result.get_stored_energy(
                        electric_fleet_unit_block, ctx.previous_time_step
                    )
                    soc = stored_energy / electric_fleet_unit_block.sizes["storage"].preexisting

                    power_flow = optimization_result.get_power_flow(electric_fleet_unit_block, ctx.previous_time_step)
                    out_power_frac = power_flow["out"] / electric_fleet_unit_block.pwr_dis_max
                    in_power_frac = power_flow["in"] / electric_fleet_unit_block.pwr_chg_max
                    real_power_unit = in_power_frac if in_power_frac > 0 else -out_power_frac
                efus_soc.append(soc)
                efus_real_power_unit.append(real_power_unit)

                required_soc_forecast = self._forecast_provider.get_efu_required_soc_forecast(
                    electric_fleet_unit_block, ctx.time, self._soc_min
                )
                efus_required_soc.append(required_soc_forecast)

                soc_diff = soc - required_soc_forecast[0]
                efus_current_soc_diff.append(soc_diff)

                efu_available_forecast = self._forecast_provider.get_efu_available_forecast(
                    electric_fleet_unit_block, ctx.time
                )
                efus_available_forecast.append(efu_available_forecast)
                efu_available_now = efu_available_forecast[0]
                efus_available_now.append(efu_available_now)
                if efu_available_now == 1.0:
                    efu_unavailable_indices = np.where(efu_available_forecast == 0.0)[0]
                    efu_next_departure_idx = (
                        efu_unavailable_indices[0]
                        if len(efu_unavailable_indices) > 0
                        else len(efu_available_forecast) - 1
                    )
                    efu_urgency = 1.0 - (efu_next_departure_idx / (len(efu_available_forecast) - 1))
                else:
                    efu_next_departure_idx = -1
                    efu_urgency = 0.0
                efus_urgency.append(efu_urgency)

                required_soc_diff = soc - required_soc_forecast[efu_next_departure_idx]
                efus_next_required_soc_diff.append(required_soc_diff)

        return {
            OBS_KEY_EFUS_SOC: np.array(efus_soc, dtype=np.float32),
            OBS_KEY_EFUS_CURRENT_SOC_DIFF: np.array(efus_current_soc_diff, dtype=np.float32),
            OBS_KEY_EFUS_NEXT_REQUIRED_SOC_DIFF: np.array(efus_next_required_soc_diff, dtype=np.float32),
            OBS_KEY_EFUS_AVAILABLE_NOW: np.array(efus_available_now, dtype=np.float32),
            OBS_KEY_EFUS_URGENCY: np.array(efus_urgency, dtype=np.float32),
            # OBS_KEY_EFUS_REQUIRED_SOCS: np.array(cars_required_socs, dtype=np.float32),
            OBS_KEY_EFUS_REAL_POWER_UNIT: np.array(efus_real_power_unit, dtype=np.float32),
            # OBS_KEY_EFUS_AVAILABLE: np.array(cars_available, dtype=np.float32),
            OBS_KEY_FLEETS_IN_POWER: np.array(fleet_in_power_units, dtype=np.float32),
            OBS_KEY_FLEETS_OUT_POWER: np.array(fleet_out_power_units, dtype=np.float32),
        }

    def get_grid_markets_features(
        self, ctx: context.Context, optimization_result: optimization.OptimizationResult | None = None
    ) -> dict[str, np.ndarray]:
        grid_markets_import_cost = []
        grid_markets_import_power_unit = []
        grid_markets_export_cost = []
        grid_markets_export_power_unit = []
        for grid_market_block in ctx.grid_market_blocks:
            import_cost = self._forecast_provider.get_grid_import_cost_forecast(grid_market_block, ctx.time)
            grid_markets_import_cost.append(import_cost)

            export_cost = self._forecast_provider.get_grid_export_cost_forecast(grid_market_block, ctx.time)
            grid_markets_export_cost.append(export_cost)

            if optimization_result is not None:
                power_flows = optimization_result.get_power_flow(grid_market_block, ctx.previous_time_step)

                import_power_max = grid_market_block.pwr_g2s
                import_power_unit = power_flows["in"] / import_power_max
                clipped_import_power_unit = np.clip(import_power_unit, 0.0, 1.0)
                grid_markets_import_power_unit.append(clipped_import_power_unit)

                export_power_max = grid_market_block.pwr_s2g
                export_power_unit = power_flows["out"] / export_power_max
                clipped_export_power_unit = np.clip(export_power_unit, 0.0, 1.0)
                grid_markets_export_power_unit.append(clipped_export_power_unit)
            else:
                grid_markets_import_power_unit.append(0.0)
                grid_markets_export_power_unit.append(0.0)

        return {
            OBS_KEY_GRID_IMPORT_COSTS: np.array(grid_markets_import_cost, dtype=np.float32),
            OBS_KEY_GRID_IMPORT_POWER: np.array(grid_markets_import_power_unit, dtype=np.float32),
            OBS_KEY_GRID_EXPORT_COSTS: np.array(grid_markets_export_cost, dtype=np.float32),
            OBS_KEY_GRID_EXPORT_POWER: np.array(grid_markets_export_power_unit, dtype=np.float32),
        }

    def get_renewable_sources_features(
        self, ctx: context.Context, optimization_result: optimization.OptimizationResult | None = None
    ) -> dict[str, np.ndarray]:
        renewable_gens_schedule = []
        renewable_gens_power = []
        for renewable_source_block in ctx.renewable_source_blocks:
            max_renewable_gen = renewable_source_block.sizes["block"].preexisting
            production_power_forecast = self._forecast_provider.get_renewable_source_power_forecast(
                renewable_source_block, ctx.time
            )
            production_power_forecast_unit = production_power_forecast / max_renewable_gen
            clipped_production_power_unit_forecast = np.clip(production_power_forecast_unit, 0.0, 1.0)
            renewable_gens_schedule.append(clipped_production_power_unit_forecast)

            if optimization_result is not None:
                power_flow = optimization_result.get_power_flow(renewable_source_block, ctx.previous_time_step)
                production_power_unit = power_flow["out"] / max_renewable_gen
                clipped_production_power_unit = np.clip(production_power_unit, 0.0, 1.0)
                renewable_gens_power.append(clipped_production_power_unit)
            else:
                renewable_gens_power.append(0.0)

        return {
            OBS_KEY_RENEWABLES_SCHEDULE: np.array(renewable_gens_schedule, dtype=np.float32),
            OBS_KEY_RENEWABLES_POWER: np.array(renewable_gens_power, dtype=np.float32),
        }

    def get_fixed_demands_features(self, ctx: context.Context) -> dict[str, np.ndarray]:
        demands_powers = []
        for demand_block in ctx.fixed_demand_blocks:
            demand_load = demand_block.flows_apriori.loc["demand", ctx.current_time_step]
            demands_powers.append(demand_load)
        return {OBS_KEY_FIXED_DEMANDS: np.array(demands_powers, dtype=np.float32)}

    def get_controllable_sources_features(
        self, ctx: context.Context, optimization_result: optimization.OptimizationResult | None = None
    ) -> dict[str, np.ndarray]:
        power_units = []
        for controllable_source_block in ctx.controllable_source_blocks:
            max_power_wh = controllable_source_block.sizes["block"].preexisting
            if optimization_result is not None:
                power_flow = optimization_result.get_power_flow(controllable_source_block, ctx.previous_time_step)
                power_unit = power_flow["out"] / max_power_wh
                clipped_power_unit = np.clip(power_unit, 0.0, 1.0)
                power_units.append(clipped_power_unit)
            else:
                power_units.append(0.0)

        return {OBS_KEY_CONTROLLABLE_SOURCES_POWER: np.array(power_units, dtype=np.float32)}

    def get_stationary_batteries_features(
        self, ctx: context.Context, optimization_result: optimization.OptimizationResult | None = None
    ) -> dict[str, np.ndarray]:
        socs = []
        for stationary_battery_block in ctx.stationary_battery_blocks:
            max_energy_wh = stationary_battery_block.sizes["storage"].preexisting
            if optimization_result is not None:
                curr_energy_wh = optimization_result.get_stored_energy(stationary_battery_block, ctx.previous_time_step)
                soc = curr_energy_wh / max_energy_wh
                clipped_soc = np.clip(soc, 0.0, 1.0)
                socs.append(clipped_soc)
            else:
                # TODO: SoC is currently always initialized to 100%, which is not realistic.
                socs.append(1.0)

        return {OBS_KEY_STATIONARY_BATTERIES_SOC: np.array(socs, dtype=np.float32)}

    def get_observation_dict(self, ctx: context.Context) -> dict[str, gym.spaces.Box]:
        obs_dict = {
            # Time information like
            OBS_KEY_TIME_FEATURES: gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            # The SoCs of the vehicles at the current time step.
            OBS_KEY_EFUS_SOC: gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(ctx.electric_fleet_unit_blocks),), dtype=np.float32
            ),
            OBS_KEY_EFUS_CURRENT_SOC_DIFF: gym.spaces.Box(
                low=-1.0, high=1.0, shape=(len(ctx.electric_fleet_unit_blocks),), dtype=np.float32
            ),
            OBS_KEY_EFUS_NEXT_REQUIRED_SOC_DIFF: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(ctx.electric_fleet_unit_blocks),),
                dtype=np.float32,
            ),
            # A forecast for each vehicle, if it is available for charging in the current and upcoming time steps.
            # OBS_KEY_EFUS_AVAILABLE: gym.spaces.Box(
            #     low=0.0,
            #     high=1.0,
            #     shape=(len(ctx.electric_fleet_unit_blocks), self._forecast_provider.forecast_horizon),
            #     dtype=np.float32,
            # ),
            OBS_KEY_EFUS_AVAILABLE_NOW: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(ctx.electric_fleet_unit_blocks),),
                dtype=np.float32,
            ),
            OBS_KEY_EFUS_URGENCY: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(ctx.electric_fleet_unit_blocks),),
                dtype=np.float32,
            ),
            # A forecast for each vehicle, of its required SoC.
            # OBS_KEY_EFUS_REQUIRED_SOCS: gym.spaces.Box(
            #     low=0.0,
            #     high=1.0,
            #     shape=(len(ctx.electric_fleet_unit_blocks), self._forecast_provider.forecast_horizon),
            #     dtype=np.float32,
            # ),
            OBS_KEY_EFUS_REAL_POWER_UNIT: gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(ctx.electric_fleet_unit_blocks),),
                dtype=np.float32,
            ),
            OBS_KEY_FLEETS_IN_POWER: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(ctx.electric_fleets.keys()),),
                dtype=np.float32,
            ),
            OBS_KEY_FLEETS_OUT_POWER: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(ctx.electric_fleets.keys()),),
                dtype=np.float32,
            ),
        }
        if ctx.has_grid_connection:
            obs_dict[OBS_KEY_GRID_IMPORT_COSTS] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(ctx.grid_market_blocks), self._forecast_provider.forecast_horizon),
                dtype=np.float32,
            )

            obs_dict[OBS_KEY_GRID_IMPORT_POWER] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(ctx.grid_market_blocks),), dtype=np.float32
            )

            obs_dict[OBS_KEY_GRID_EXPORT_COSTS] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(ctx.grid_market_blocks), self._forecast_provider.forecast_horizon),
                dtype=np.float32,
            )

            obs_dict[OBS_KEY_GRID_EXPORT_POWER] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(ctx.grid_market_blocks),), dtype=np.float32
            )

        if ctx.has_renewable_sources:
            obs_dict[OBS_KEY_RENEWABLES_SCHEDULE] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(ctx.renewable_source_blocks), self._forecast_provider.forecast_horizon),
                dtype=np.float32,
            )
            obs_dict[OBS_KEY_RENEWABLES_POWER] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(ctx.renewable_source_blocks),),
                dtype=np.float32,
            )

        if ctx.has_stationary_batteries:
            obs_dict[OBS_KEY_STATIONARY_BATTERIES_SOC] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(ctx.stationary_battery_blocks),), dtype=np.float32
            )

        if ctx.has_controllable_sources:
            # The current generation of each controllable source.
            obs_dict[OBS_KEY_CONTROLLABLE_SOURCES_POWER] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(ctx.controllable_source_blocks),),
                dtype=np.float32,
            )

        if ctx.has_fixed_demands:
            # The current consumption of each load.
            obs_dict[OBS_KEY_FIXED_DEMANDS] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(ctx.fixed_demand_blocks),),
                dtype=np.float32,
            )

        return obs_dict
