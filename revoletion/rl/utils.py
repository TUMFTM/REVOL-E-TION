import numpy as np
import pandas as pd

from revoletion import blocks, utils


def get_soc_envelope(
    block: blocks.ElectricFleetUnit,
    horizon: utils.TimeSettings,
    dsoc_step_max: float | None = None,
    dsoc_step_min: float | None = None,
    target_soc: float | None = None,
) -> pd.Series:
    dti = horizon.dti

    plugged = block.log.loc[dti, "atbase"]

    nom_capacity_wh = block.sizes["storage"].preexisting

    max_charge_power_w = block.pwr_chg_max * block.eff["chg_int"] * np.sqrt(block.eff["storage_roundtrip"])
    if dsoc_step_max is None:
        dsoc_step_max = (max_charge_power_w * horizon.timestep.hours) / nom_capacity_wh
    else:
        dsoc_step_max = dsoc_step_max

    if dsoc_step_min is None:
        dsoc_step_min = (max_charge_power_w * 0.1 * horizon.timestep.hours) / nom_capacity_wh
    else:
        dsoc_step_min = dsoc_step_min

    consumption = block.log.loc[dti, "consumption"] * horizon.timestep.hours
    # Need at least enough SoC to compensate standing loss.
    consumption += block.loss_rate_per_ts * nom_capacity_wh

    dsoc = consumption / nom_capacity_wh

    soc_floor = pd.Series(0.0, index=dti, dtype=np.float64)

    required_soc = target_soc or 0.0

    for time_step in reversed(dti):
        required_soc = min(required_soc + dsoc[time_step], 1.0)

        if plugged[time_step]:
            if required_soc > dsoc_step_min:
                required_soc = max(required_soc - dsoc_step_max, 0.0)

        soc_floor[time_step] = required_soc

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
    max_charge_power_w = block.pwr_chg_max * block.eff["chg_int"] * np.sqrt(block.eff["storage_roundtrip"])
    dsoc_step_max = (max_charge_power_w * horizon.timestep.hours) / nom_capacity_wh

    consumption = block.log.loc[horizon.dti, "consumption"] * horizon.timestep.hours
    # Need at least enough SoC to compensate standing loss.
    consumption += block.loss_rate_per_ts * nom_capacity_wh

    dsoc = consumption / nom_capacity_wh

    # Calculate power required at each time step
    for i in range(len(horizon.dti) - 1):
        current_time_step = horizon.dti[i]
        next_time_step = horizon.dti[i + 1]
        current_soc -= dsoc[current_time_step]

        if plugged[current_time_step]:
            required_dsoc = max(soc_envelope[next_time_step] - current_soc, 0.0)
            charge_dsoc = min(required_dsoc, dsoc_step_max)
            power_required = (
                (charge_dsoc * nom_capacity_wh)
                / timestep_hours
                / block.eff["chg_int"]
                / np.sqrt(block.eff["storage_roundtrip"])
            )

            if power_required > 0.0:
                current_soc += required_dsoc
                power_envelope[current_time_step] = power_required

    return power_envelope
