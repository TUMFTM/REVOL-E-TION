# Aging Model // Paper Charging Strategy x Aging
#Input: Power-Profile (Time-Series)
#Output: Capacity-Loss for Aging-Timeframe

from ElectroThermalSingle import calcElectroThermalValuesSingle
import numpy as np

#Debug
import matplotlib.pyplot as plt

## Initialize Result-Arrays

p_value_control = np.zeros((len(p_profile)))
SOC = np.zeros((len(p_profile)+1))
Crate = np.zeros((len(p_profile)))
P_Loss = np.zeros((len(p_profile)))
T_Cell = np.zeros((len(p_profile)+1))
T_Housing= np.zeros((len(p_profile)+1))
P_Cool = np.zeros((len(p_profile)))
P_Heat = np.zeros((len(p_profile)))

## Initialize Start Values
T_Cell[0] = 20  # Start Temperature of cells
T_Housing[0] = 20  # Start Temperature of Housing
SOC[0] = cell.SOC_max  # Start SOC of cells
T_amb = 20

for idx_p_profile, P_Value_Vehicle in enumerate(p_profile[0:60000]):

    SOC[idx_p_profile+1], Crate[idx_p_profile], P_Value_control, P_Cool[idx_p_profile], T_Cell[idx_p_profile+1], T_Housing[idx_p_profile+1] = calcElectroThermalValuesSingle(cell, bet,
                                        n_cells, T_Cell[idx_p_profile], SOC[idx_p_profile], Crate[idx_p_profile], P_Value_Vehicle, P_Cool[idx_p_profile-1], T_Housing[idx_p_profile], T_amb, dt)

plt.plot(SOC)
plt.show()


# Aging Model
# Input: SOC / T_Cell / Crate / Cell data

#Input previous calculated Aging Results
Q_loss_cal_prev = 0
Q_loss_cyc_prev = 0
R_inc_cal_prev = 0
R_inc_cyc_prev = 0

res_aging = calc_aging_routing(cell_chemistry, cell, SOC, T_Cell, Crate, Q_loss_cal_prev, Q_loss_cyc_prev, R_inc_cal_prev, R_inc_cyc_prev, dt)

print(res_aging[0])

