

from Thermal_Model import thermal_control
from Thermal_Model import thermal_model

def calcElectroThermalValuesSingle(cell,bet, n_cells, T_Cell, SOC, Crate, P_Value_Vehicle, P_Cool_prev, T_Housing, T_amb, dt):

    #Scale vehicle power demand to cell power demand
    P_Value_Cell = P_Value_Vehicle / n_cells

    #CCCV + Prevent Overcharge
    P_Value_control = electric_control(cell, P_Value_Cell, SOC, T_Cell, Crate, dt)

    #Heating / Cooling
    P_Value_control, P_Cool, P_Heat = thermal_control(bet, T_Cell, P_Cool_prev, P_Value_control, P_Value_Cell, n_cells)

    #New Electric State
    SOC_new, Crate_new, P_Loss = electric_model(cell, T_Cell, SOC, P_Value_control, dt)

    # New Thermal State
    T_Cell_new, T_Housing_new = thermal_model(bet, cell, n_cells, T_Cell, T_Housing, T_amb, P_Loss, P_Cool, P_Heat, dt)

    return SOC_new, Crate_new, P_Value_control, P_Cool, T_Cell_new, T_Housing_new