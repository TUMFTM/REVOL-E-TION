import abc

import numpy as np
import pandas as pd
import typing_extensions

from revoletion import blocks

from . import _context as context
from . import utils as rl_utils


class ForecastProvider(abc.ABC):
    _forecast_horizon: int

    def __init__(self, forecast_horizon: int) -> None:
        self._forecast_horizon = forecast_horizon

    @property
    def forecast_horizon(self) -> int:
        return self._forecast_horizon

    def get_forecast(
        self, time_series: pd.DataFrame, time_ctx: context.TimeContext, pad_val: float = 0.0
    ) -> np.ndarray:
        forecast_horizon_dti = time_ctx.horizon.dti[time_ctx.step_idx : time_ctx.step_idx + self._forecast_horizon]
        forecast_values = time_series.loc[forecast_horizon_dti].values

        return np.pad(forecast_values, (0, self._forecast_horizon - len(forecast_values)), constant_values=pad_val)

    @abc.abstractmethod
    def get_renewable_source_power_forecast(
        self, pv_block: blocks.PVSource, time_ctx: context.TimeContext
    ) -> np.ndarray: ...

    @abc.abstractmethod
    def get_grid_export_cost_forecast(
        self, grid_market_block: blocks.GridMarket, time_ctx: context.TimeContext
    ) -> np.ndarray: ...

    @abc.abstractmethod
    def get_grid_import_cost_forecast(
        self, grid_market_block: blocks.GridMarket, time_ctx: context.TimeContext
    ) -> np.ndarray: ...

    @abc.abstractmethod
    def get_efu_required_soc_forecast(
        self, efu_block: blocks.ElectricFleetUnit, time_ctx: context.TimeContext, soc_min: float = 0.0
    ) -> np.ndarray: ...

    @abc.abstractmethod
    def get_efu_available_forecast(
        self, efu_block: blocks.ElectricFleetUnit, time_ctx: context.TimeContext
    ) -> np.ndarray: ...


class PerfectForesightForecastProvider(ForecastProvider):
    @typing_extensions.override
    def get_renewable_source_power_forecast(
        self, pv_block: blocks.PVSource, time_ctx: context.TimeContext
    ) -> np.ndarray:
        power_spec = pv_block.data["power_spec"]
        return self.get_forecast(power_spec, time_ctx)

    @typing_extensions.override
    def get_grid_export_cost_forecast(
        self, grid_market_block: blocks.GridMarket, time_ctx: context.TimeContext
    ) -> np.ndarray:
        export_cost = grid_market_block.evaluators["s2g"].opt.spec_ep_operation
        return self.get_forecast(export_cost, time_ctx)

    @typing_extensions.override
    def get_grid_import_cost_forecast(
        self, grid_market_block: blocks.GridMarket, time_ctx: context.TimeContext
    ) -> np.ndarray:
        import_cost = grid_market_block.evaluators["g2s"].opt.spec_ep_operation
        return self.get_forecast(import_cost, time_ctx)

    @typing_extensions.override
    def get_efu_required_soc_forecast(
        self, efu_block: blocks.ElectricFleetUnit, time_ctx: context.TimeContext, soc_min: float = 0.0
    ) -> np.ndarray:
        soc_envelope = rl_utils.get_soc_envelope(efu_block, time_ctx.horizon) + soc_min
        soc_envelope = np.clip(soc_envelope, 0.0, 1.0)
        return self.get_forecast(soc_envelope, time_ctx, pad_val=soc_min)

    @typing_extensions.override
    def get_efu_available_forecast(
        self, efu_block: blocks.ElectricFleetUnit, time_ctx: context.TimeContext
    ) -> np.ndarray:
        atbase = efu_block.log["atbase"]
        return self.get_forecast(atbase, time_ctx).astype(np.float32)


class LimitedForecastProvider(ForecastProvider):
    def __init__(self, forecast_horizon: int, rng: np.random.Generator | None = None) -> None:
        super().__init__(forecast_horizon)
        self._rng = rng or np.random.default_rng()

    @typing_extensions.override
    def get_renewable_source_power_forecast(
        self, pv_block: blocks.PVSource, time_ctx: context.TimeContext
    ) -> np.ndarray:
        power_spec = pv_block.data["power_spec"]
        power_spec_forecast = self.get_forecast(power_spec, time_ctx)

        noisy_forecast = self.sqrt_uncertainty(power_spec_forecast, self._forecast_horizon)
        return np.clip(noisy_forecast, 0.0, 1.0)

    @typing_extensions.override
    def get_grid_export_cost_forecast(
        self, grid_market_block: blocks.GridMarket, time_ctx: context.TimeContext
    ) -> np.ndarray:
        export_cost = grid_market_block.evaluators["s2g"].opt.spec_ep_operation
        export_cost_forecast = self.get_forecast(export_cost, time_ctx)
        noisy_forecast = self.sqrt_uncertainty(export_cost_forecast, self._forecast_horizon, base_std=5e-2, eps=-1e-6)
        return noisy_forecast

    @typing_extensions.override
    def get_grid_import_cost_forecast(
        self, grid_market_block: blocks.GridMarket, time_ctx: context.TimeContext
    ) -> np.ndarray:
        import_cost = grid_market_block.evaluators["g2s"].opt.spec_ep_operation
        import_cost_forecast = self.get_forecast(import_cost, time_ctx)
        noisy_forecast = self.sqrt_uncertainty(import_cost_forecast, self._forecast_horizon, base_std=5e-2, eps=1e-6)
        return noisy_forecast

    @typing_extensions.override
    def get_efu_required_soc_forecast(
        self, efu_block: blocks.ElectricFleetUnit, time_ctx: context.TimeContext, soc_min: float = 0.0
    ) -> np.ndarray:
        soc_envelope_horizon = time_ctx.horizon.cut(
            time_ctx.step_idx, min(self._forecast_horizon, len(time_ctx.horizon) - time_ctx.step_idx - 1)
        )
        soc_envelope = (rl_utils.get_soc_envelope(efu_block, soc_envelope_horizon) + soc_min)[
            0 : self._forecast_horizon
        ]
        padded_soc_envelope = np.pad(
            soc_envelope.values,
            (0, self._forecast_horizon - len(soc_envelope)),
            constant_values=soc_min,
        )
        return padded_soc_envelope

    @typing_extensions.override
    def get_efu_available_forecast(
        self, efu_block: blocks.ElectricFleetUnit, time_ctx: context.TimeContext
    ) -> np.ndarray:
        atbase = efu_block.log["atbase"]
        return self.get_forecast(atbase, time_ctx).astype(np.float32)

    def sqrt_uncertainty(self, forecast, time_steps, base_std=0.05, eps: float = 1e-6):
        """
        Uncertainty grows with square root of time horizon.
        Models random walk behavior - common in forecasting.
        """
        std_schedule = base_std * np.sqrt(np.arange(1, time_steps + 1))
        noise = self._rng.normal(0, 1, time_steps)
        uncertainty = noise * std_schedule * (forecast + eps)
        return forecast + uncertainty

    def auto_regressiv_uncertainty(self, forecast, time_steps, initial_std=0.02, final_std=0.20, ar_coef=0.7):
        std_schedule = np.linspace(initial_std, final_std, time_steps)
        noise = np.zeros(time_steps)
        noise[0] = self._rng.normal(0, std_schedule[0])

        for t in range(1, time_steps):
            noise[t] = ar_coef * noise[t - 1] + np.random.normal(0, std_schedule[t] * np.sqrt(1 - ar_coef**2))

        uncertainty = noise * (forecast + 1e-3)
        return forecast + uncertainty
