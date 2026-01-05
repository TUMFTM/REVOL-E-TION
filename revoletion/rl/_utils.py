import numpy as np
import pandas as pd

from revoletion import blocks, utils


def get_soc_envelope(
    block: blocks.ElectricFleetUnit, horizon: utils.TimeSettings, dsoc_step: float | None = None
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

    required_soc = 0.0

    for time_step in reversed(horizon.dti):
        required_soc += dsoc[time_step]

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
    # Get the SoC envelope first

    # Initialize power envelope
    power_envelope = pd.Series(0.0, index=horizon.dti, dtype=np.float64)

    # Get necessary parameters
    plugged = block.log.loc[horizon.dti, "atbase"]
    nom_capacity_wh = block.sizes["storage"].preexisting
    timestep_hours = horizon.timestep.hours

    # Calculate power required at each time step
    for i in range(len(horizon.dti) - 1):
        current_time = horizon.dti[i]
        next_time = horizon.dti[i + 1]

        if plugged[current_time]:
            # Calculate the change in SoC needed
            dsoc = soc_envelope[next_time] - soc_envelope[current_time]

            # Convert to power (W)
            # Positive dsoc means SoC needs to increase (charging)
            power_required = (dsoc * nom_capacity_wh) / timestep_hours

            # Account for charging efficiency (power from grid)
            if power_required > 0:
                power_required = power_required / block.eff["chg_int"]

            power_envelope[current_time] = max(power_required, 0.0)
        else:
            # Not plugged in, cannot charge
            power_envelope[current_time] = 0.0

    # Last time step has no "next" so set to 0
    power_envelope[horizon.dti[-1]] = 0.0

    return power_envelope
