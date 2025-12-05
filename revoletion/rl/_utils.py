import numpy as np
import pandas as pd

from revoletion import blocks, utils


def get_soc_envelope(block: blocks.ElectricFleetUnit, horizon: utils.TimeSettings) -> pd.Series:
    plugged = block.log.loc[horizon.dti, "atbase"]

    nom_capacity_wh = block.sizes["storage"].preexisting

    max_charge_power_w = block.pwr_chg_max * block.eff["chg_int"]
    dsoc_step_max = (max_charge_power_w * horizon.timestep.hours) / nom_capacity_wh

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
