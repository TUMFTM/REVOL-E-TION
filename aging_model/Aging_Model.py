import numpy as np



def calc_aging_routing(cell_chemistry, cell, SOC, T_Cell, Crate,Q_loss_cal_prev, Q_loss_cyc_prev, R_inc_cal_prev, R_inc_cyc_prev, dt):
    # Choice of Aging Model (NMC/LFP)

    if cell_chemistry == 0:
        res_aging = calc_aging_schmalstieg(cell, T_Cell, SOC, Crate, Q_loss_cal_prev, Q_loss_cyc_prev, R_inc_cyc_prev, R_inc_cal_prev, dt)
    if cell_chemistry == 1:
        res_aging = calc_aging_naumann(cell, T_Cell, SOC, Crate, Q_loss_cal_prev, Q_loss_cyc_prev, R_inc_cal_prev, R_inc_cyc_prev, dt)
    else:
        raise

    return res_aging



def calc_aging_naumann(cell, T_Cell, SOC, Crate, Q_loss_cal_prev, Q_loss_cyc_prev, R_inc_cal_prev, R_inc_cyc_prev, dt):
    # Naumann et al. //

    ## -- Calculation Stress - Factors

    # Pre-Calculation
    T_Cell = T_Cell + 273.15

    #  Calendar Ageing
    k_temp_Q_cal = np.mean(1.2571e-05 * np.exp((-17126 / 8.3145) * (1 / T_Cell - 1 / 298.15)))
    k_temp_R_cal = np.mean(3.419e-10 * np.exp((-71827 / 8.3145) * (1 / T_Cell - 1 / 298.15)))

    k_soc_Q_cal = np.mean(2.85750 * ((SOC - 0.5) ** 3) + 0.60225)
    k_soc_R_cal = np.mean(3.3903 * ((SOC - 0.5) ** 2) + 1.56040)


    # Cycling Aging
    # DOD Calculation

    DODs, SOCavgs = rfcounting(SOC)
    fec = sum(DODs)
    DODs = np.asarray(DODs)
    k_DOD_Q_cyc = sum((4.0253 * ((DODs - 0.6) ** 3) + 1.09230) * DODs) / fec
    k_DOD_R_cyc = sum((6.8477 * ((DODs - 0.5) ** 3) + 0.91882) * DODs) / fec

    Caging = abs(Crate)
    k_C_Q_cyc = sum(0.0971 + 0.063 * Caging) / len(Caging)
    k_C_R_cyc = sum(0.0023 - 0.0018 * Caging) / len(Caging)

    ## -- Calculation Q_Loss

    # Time in Seconds!
    t_aging = len(SOC) * dt  # Time for which aging is evaluated

    t_eq = (Q_loss_cal_prev / k_soc_Q_cal / k_temp_Q_cal) ** 2
    Qloss_new_cal = k_temp_Q_cal * k_soc_Q_cal * np.sqrt(t_eq + t_aging)
    Rinc_new_cal = R_inc_cal_prev + k_temp_R_cal * k_soc_R_cal * t_aging

    FEC_eq = (100 * Q_loss_cyc_prev / k_DOD_Q_cyc / k_C_Q_cyc) ** 2
    Qloss_new_cyc = 0.01 * (k_DOD_Q_cyc * k_C_Q_cyc) * np.sqrt(FEC_eq + fec)
    Rinc_new_cyc = R_inc_cyc_prev + 0.01 *(k_DOD_R_cyc * k_C_R_cyc) * fec

    Qloss_ges = Qloss_new_cal + Qloss_new_cyc
    Rinc_ges = Rinc_new_cal + Rinc_new_cyc

    return [Qloss_ges, Rinc_ges, Qloss_new_cal,Rinc_new_cal, Qloss_new_cyc, Rinc_new_cyc]


def calc_aging_schmalstieg(cell, T_Cell, SOC, Crate, Q_loss_cal_prev, Q_loss_cyc_prev, R_inc_cyc_prev, R_inc_cal_prev, dt):
    # Schmalstieg et al. // Calculation Stress-Factors

    ## -- Calculation Stress - Factors

    # Pre-Calculation
    k_VW = 0.43
    T_Cell = T_Cell + 273.15
    T_Cell[
        T_Cell < 298.15] = 298.15  # Modification: Calendar aging is constant below 25°C, where the aging model is not valid

    # Calendar Aging
    Uocvs = cell.ocv_func(SOC)
    k_temp_Q_cal = np.mean(1e6 * np.exp(-6976 / T_Cell))
    k_temp_R_cal = np.mean(1e5 * np.exp(-5986 / T_Cell))

    k_soc_Q_cal = np.mean(7.543 * Uocvs - 23.75)
    k_soc_R_cal = np.mean(5.27 * Uocvs - 16.32)

    # Cycling Aging

    # DOD Calculation
    DODs, SOCavgs = rfcounting(SOC)
    Uavgs = cell.ocv_func(SOCavgs)

    Qtot = sum(DODs) * cell.Qnom
    DODs = np.asarray(DODs)

    k_DOD_Q_cyc = sum((4.081e-3 * DODs) * DODs * cell.Qnom) / Qtot
    k_DOD_R_cyc = sum((2.798e-4 * DODs) * DODs * cell.Qnom) / Qtot

    k_Uavgs_Q_cyc = sum((7.348e-3 * (Uavgs - 3.667) ** 2 + 7.6e-4) * DODs * cell.Qnom) / Qtot
    k_Uavgs_R_cyc = sum((2.153e-4 * (Uavgs - 3.725) ** 2 - 1.521e-5) * DODs * cell.Qnom) / Qtot


    ## -- Calculation Q_Loss


    Q_tot_age = Qtot * 2
    Q_tot_eq = (Q_loss_cyc_prev / k_VW / (k_DOD_Q_cyc + k_Uavgs_Q_cyc)) ** 2
    Qloss_new_cyc = k_VW * (k_DOD_Q_cyc + k_Uavgs_Q_cyc) * np.sqrt(Q_tot_eq + Q_tot_age)  # Scaled according to Teichert!
    Rinc_new_cyc = k_VW * (k_DOD_R_cyc + k_Uavgs_R_cyc) * (Q_tot_age)

    #Time = Days
    t_aging = len(SOC)*dt/3600/24 # Time for which aging is evaluated
    t_eq_Q = (Q_loss_cal_prev/ k_VW / k_temp_Q_cal / k_soc_Q_cal) ** (4 / 3)
    t_eq_R = (R_inc_cal_prev / k_VW / k_temp_R_cal / k_soc_R_cal) ** (4 / 3)

    Qloss_new_cal = k_VW * k_temp_Q_cal * k_soc_Q_cal * ((t_eq_Q + t_aging) ** 0.75)
    Rinc_new_cal = k_VW * k_temp_R_cal * k_soc_R_cal * ((t_eq_R + t_aging) ** 0.75)

    Qloss_ges = Qloss_new_cal+Qloss_new_cyc
    Rinc_ges = Rinc_new_cal+Rinc_new_cyc

    return [Qloss_ges, Rinc_ges, Qloss_new_cal,Rinc_new_cal, Qloss_new_cyc, Rinc_new_cyc]


def rfcounting(SOC):
    # Remove Over-Charged phases
    SOC[SOC > SOC[0]] = SOC[0]

    # Remove constant SOC phases
    SOC_nozero = SOC[np.concatenate(([True], np.diff(SOC) != 0))]

    # Slice to start and end with the maximum SOC
    I = np.argmax(SOC_nozero)
    SOC_sorted = np.concatenate((SOC_nozero[I:], SOC_nozero[:I]))

    # Find extremas
    slope = np.diff(SOC_sorted)
    is_extremum = np.concatenate(([True], (slope[1:] * slope[:-1]) < 0, [True]))
    SOCs = SOC_sorted[is_extremum]

    # Find DODs
    DODs = []
    SOCavgs = []
    index = 1
    while len(SOCs) > index + 1:
        prevDOD = abs(SOCs[index] - SOCs[index - 1])
        nextDOD = abs(SOCs[index] - SOCs[index + 1])
        if nextDOD < prevDOD:
            index += 1
        else:
            DODs.append(prevDOD)
            SOCavgs.append((SOCs[index] + SOCs[index - 1]) / 2)
            SOCs = np.delete(SOCs, [index - 1, index])
            index = 1
    return DODs, SOCavgs
