
import numpy as np


def electric_model(cell, T_Cell, SOC, P_Value, dt):

    R_i_ch = cell.R_i_ch_func(T_Cell, SOC)
    R_i_dch = cell.R_i_dch_func(T_Cell, SOC)
    Uocv = cell.ocv_func(SOC)

    if P_Value > 0:
        R_i = R_i_ch
    else:
        R_i = R_i_dch

    Ibat = np.real((-Uocv + np.sqrt((Uocv ** 2) + (4 * R_i * P_Value))) / (2 * R_i))
    C_rate = Ibat / (cell.Qnom)
    SOC_new = SOC + (C_rate * dt / 3600)
    P_Loss = (Ibat ** 2) * R_i


    return  SOC_new, C_rate,  P_Loss



def electric_control(cell, P_Value, SOC, T_Cell, Crate, dt):

    uccv = 0

    #CCCV - Charging
    if P_Value > 0 and uccv == 0: # Charging
        R_i_ch = cell.R_i_ch_func(T_Cell, SOC)
        Uocv = cell.ocv_func(SOC)
        Ibat = np.real((-Uocv + np.sqrt((Uocv ** 2) + (4 * R_i_ch * P_Value))) / (2 * R_i_ch))
        U_act = Uocv + Ibat * R_i_ch

        if U_act >= cell.Umax: # Derate to reach Constant Voltage
            for i in range(1000):
                if Uocv + Ibat*(1-0.001*i) * R_i_ch < cell.Umax:
                    P_Value = Ibat*(1-0.001*i) * U_act
                    Crate = Ibat*(1-0.001*i) / (cell.Qnom)
                    uccv = 1
                    break


    #Prevent Overcharge // Flag if not enough energy for next P_Value
    if P_Value > 0:
        if (SOC + (Crate * dt / 3600)) > cell.SOC_max:
             P_Value = 0

    if P_Value < 0:
        if (SOC + (Crate * dt / 3600)) < cell.SOC_min:
            check_SOC = 1 # Battery Capacity is not feasible
            #TODO: Unsinnige SOC Abfangen
    else:
        pass

    return P_Value