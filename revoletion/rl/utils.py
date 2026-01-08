import numpy as np
import pandas as pd

from revoletion import blocks, utils


def get_soc_envelope(
    block: blocks.ElectricFleetUnit,
    horizon: utils.TimeSettings,
    dsoc_step: float | None = None,
    target_soc: float | None = None,
) -> pd.Series:
    plugged = block.log.loc[horizon.dti, "atbase"]

    nom_capacity_wh = block.sizes["storage"].preexisting

    max_charge_power_w = block.pwr_chg_max * block.eff["chg_int"]
    if dsoc_step is None:
        dsoc_step_max = (max_charge_power_w * horizon.timestep.hours) / nom_capacity_wh
    else:
        dsoc_step_max = dsoc_step

    consumption = block.log.loc[horizon.dti, "consumption"] * horizon.timestep.hours
    # Need at least enough SoC to compensate standing loss.
    consumption += block.loss_rate_per_ts

    dsoc = consumption / nom_capacity_wh

    soc_floor = pd.Series(0.0, index=horizon.dti, dtype=np.float64)

    required_soc = target_soc or 0.0

    for time_step in reversed(horizon.dti):
        required_soc = min(required_soc + dsoc[time_step], 1.0)

        soc_floor[time_step] = required_soc
        if plugged[time_step]:
            required_soc = max(required_soc - dsoc_step_max, 0.0)

    return soc_floor


def get_power_envelope(
    block: blocks.ElectricFleetUnit, horizon: utils.TimeSettings, soc_envelope: pd.Series
) -> pd.Series:
    """
    Convert SoC envelope to minimum charging power required at each time step.

    Returns:
        Series of minimum charging power in Watts (positive = charging required)
    """
    # Initialize power envelope
    power_envelope = pd.Series(0.0, index=horizon.dti, dtype=np.float64)

    # Get necessary parameters
    plugged = block.log.loc[horizon.dti, "atbase"]
    current_soc = block.states.loc[horizon.start, "soc"]
    nom_capacity_wh = block.sizes["storage"].preexisting
    timestep_hours = horizon.timestep.hours

    # Calculate power required at each time step
    for i in range(len(horizon.dti) - 1):
        current_time = horizon.dti[i]
        next_time = horizon.dti[i + 1]

        if plugged[current_time]:
            required_dsoc = max(soc_envelope[next_time] - current_soc, 0.0)

            if required_dsoc > 0.0:
                # Convert to power (W)
                # Positive dsoc means SoC needs to increase (charging)
                power_required = (required_dsoc * nom_capacity_wh) / timestep_hours / block.eff["chg_int"]

                power_envelope[current_time] = max(power_required, 0.0)
                current_soc += required_dsoc
            else:
                power_envelope[current_time] = 0.0
        else:
            current_soc = max(current_soc - (soc_envelope[current_time] - soc_envelope[next_time]), 0.0)
            # Not plugged in, cannot charge
            power_envelope[current_time] = 0.0

    return power_envelope
