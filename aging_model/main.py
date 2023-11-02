# Aging Model // Paper Charging Strategy x Aging
#Input: Power-Profile (Time-Series)
#Output: Capacity-Loss for Aging-Timeframe


from init_battery_data import load_cell_data
from batteryElectricTruck import BatteryElectricTruck
from ElectroThermalSingle import calcElectroThermalValuesSingle
import numpy as np
from Aging_Model import calc_aging_routing

#Debug
import matplotlib.pyplot as plt


#Parameterization:

cell_chemistry = 1 # LFP = 0 // Schmalstieg = 1
battery_capacity = 500 # in kWh
bet = BatteryElectricTruck()
dt = 1 # Simulation Step-Time in s

# Import Test Power Profile
p_profile = np.genfromtxt("Test_File/betos_power_profile.csv", delimiter='\t')
p_profile= p_profile*(-1) # - = discharge // + = charge
### Select Chemistry/Models ------------------------------------------
# Chemistry // 0:Schmalstieg et. al. // 1:Naumann et.al.
cell = load_cell_data(cell_chemistry, bet) #Initialize Cell Data + Gravimetric Density / C2P
n_cells = (battery_capacity * 1000) / (cell.Qnom * cell.Unom) # Number of Cells // For scaling to single cell




### Single Call -----------------------------------------------------

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
# Output: [Qloss_ges, Rinc_ges, Qloss_new_cal,Rinc_new_cal, Qloss_new_cyc, Rinc_new_cyc]

