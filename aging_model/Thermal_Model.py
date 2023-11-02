import numpy as np

def thermal_model(bet, cell, n_cells, T_Cell, T_Housing, T_amb, P_Loss, Pcool, Pheat, dt):

    #Mass Calculation based on battery size
    m_battery_wo_cells = ((n_cells * cell.Qnom * cell.Unom) / cell.grav_energy_density) * (1-cell.c2p_grav)
    Cth_Battery = cell.mass * 1045 # J/(kgK) OlafDiss

    Cth_Housing = bet.c_aluminium * m_battery_wo_cells # J/K

    T_Housing_new = T_Housing + dt * (((Pcool * bet.COPcool) + (Pheat * bet.COPheat) + bet.k_bh * n_cells * (T_Cell -
                                                        T_Housing) + bet.k_out * (T_amb - T_Housing)) / Cth_Housing)

    T_Cell_new = T_Cell + dt * ((P_Loss + bet.k_bh * (T_Housing - T_Cell)) / Cth_Battery)

    return T_Cell_new, T_Housing_new


def thermal_control(bet, T_Cell, p_cool_prev, p_value_control, p_value, n_cells):

   # Control Algorithm for active Cooling
    if T_Cell < bet.T_Heat:
        P_Heat = bet.Pheater
        P_Cool = 0
    elif T_Cell > bet.T_Cool_on or (T_Cell > bet.T_Cool_off and p_cool_prev > 0): # Cool if Cooling-Threshold is exeeded or (if Cooling was active in the previous time step and Off-Threshold is not reached yet)
        P_Heat = 0
        P_Cool = bet.Pcooler
    else:
        P_Heat = 0
        P_Cool = 0

    #Impact of Cooling on Power

    #Case Driving // Cooling Power added to Power demand of driving task
    if p_value <= 0:
        p_value_control = p_value_control + (P_Cool+P_Heat)/n_cells

    # Case Charging // Cooling Power from Infrastructure -> If Cell is limiting no further power demand from cooling
    else:
        if p_value >= p_value_control:
            p_value_control = p_value_control - (P_Cool + P_Heat) / n_cells

    return p_value_control, P_Cool, P_Heat
