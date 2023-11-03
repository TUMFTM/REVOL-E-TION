import numpy as np
import scipy.interpolate as spip
import pickle
import os

class BatteryPack:

    def __init__(self, chemistry, size):
        # Battery Thermal System
        self.chemistry = chemistry.lower()  # 'lfp' or 'nmc'
        self.size = size  # initial nominal capacity in Wh

        self.soc_max = 1  # Upper SOC limit
        self.soc_min = 0  # Lower SOC limit

        self.c_th = 76.27  # Cell heat capacity as per Forgez et al. # todo find unit
        self.R_th_in = 3.3  # Thermal resistance between cell and Housing # todo find unit

        self.q_loss_cal = self.r_inc_cal = self.q_loss_cyc = self.r_inc_cyc = 0  # Initial values for state tracking

        if self.chemistry == 'nmc':
            # Cell from Schmalstieg et al. - Sanyo UR18650E
            self.q_nom = 2.15  # Typical capacity in Ah
            self.u_nom = 3.6  # Nominal voltage in V
            self.u_min = 3.0  # Minimum voltage in V
            self.u_max = 4.2  # Maximum voltage in V
            self.i_max_cont = 2.05  # Maximum charging current in A
            self.i_min_cont = -6.15  # Maximum discharging current in A
            self.m = 0.0445  # Cell mass in kg
            self.data_path = os.path.join(os.getcwd(), 'input', 'battery', 'input/battery/sanyo_ur18650e.pkl')

        elif self.chemistry == 'lfp':
            # Cell from Naumann et al. - Sony US26650
            self.q_nom = 3.0  # Typical capacity in Ah
            self.u_nom = 3.2  # Nominal voltage in V
            self.u_min = 2.0  # Minimum voltage in V
            self.u_max = 3.6  # Maximum voltage in V
            self.i_max_cont = 3  # Maximum charging current in A
            self.i_min_cont = -20  # Maximum discharging current in A
            self.m = 0.0845  # Cell mass in kg
            self.data_path = os.path.join(os.getcwd(), 'input', 'battery', 'input/battery/sony_us26650.pkl')

        with open(self.data_path, 'rb') as file:
            self.ocv, self.r_i_ch, self.r_i_dch = pickle.load(file)

        self.ocv_interp = spip.RegularGridInterpolator(points=(self.ocv.index.to_list(),
                                                               self.ocv.colums.to_list()),
                                                       values=self.ocv.to_numpy(),
                                                       method='linear')
        self.r_i_ch_interp = spip.RegularGridInterpolator(points=(self.r_i_ch.index.to_list(),
                                                                  self.r_i_ch.columns.to_list()),
                                                          values=self.r_i_ch.to_numpy(),
                                                          method='linear')
        self.r_i_dch_interp = spip.RegularGridInterpolator(points=(self.r_i_dch.index.to_list(),
                                                                   self.r_i_dch.columns.to_list()),
                                                           values=self.r_i_dch.to_numpy(),
                                                           method='linear')

        self.e_cell = self.q_nom * self.u_nom
        self.n_cells = self.size / self.e_cell  # number of cells is a float here to correctly represent power split

        self.p_cool = 10e3  # Cooling power in W [Schimpe et al.]
        self.p_heat = 11.2e3  # Heating power in W [Schimpe et al.]
        self.cop_cool = -3  # Coefficient of performance of cooling system in pu [Schimpe et al.]
        self.cop_heat = 4  # Coefficient of performance of heating system in pu [Schimpe et al., Danish Energy Agency 2012]
        self.Ebat_Neubauer = 22.1e3  # Battery size used by Neubauer et al. in Wh [Neubauer et al.]
        self.Rth_Neubauer = 4.343  # Thermal resistance between housing and ambient used by Neubauer et al. in W/K [Neubauer et al.]
        self.Cth_Neubauer = 182e3  # Thermal mass of battery in J/K [Neubauer]
        self.T_Cool_on = 33  # Activation Cooling Threshold [Neubauer et al.]
        self.T_Cool_off = 31.5  # Deactivation Cooling Threshold [ID3 Paper]
        self.T_Heat = 10  # Heating Threshold [Neubauer et al.]
        self.k_bh = 0.899  # K/W #OlafDiss
        self.c_aluminium = 896  # J/(kgK)#OlafDiss
        self.k_out = 10.9  # W/K OlafDiss
        self.gravimetric_energy_density = [273, 176]  # in Wh/kg
        self.volumetric_energy_density = [685, 376]  # in Wh/l
        self.c2p_grav = [0.59, 0.71]
        self.c2p_vol = [0.39, 0.55]

        # Import Test Power Profile
        p_profile = np.genfromtxt("Test_File/betos_power_profile.csv", delimiter='\t')
        p_profile = p_profile * (-1)  # - = discharge // + = charge
        ### Select Chemistry/Models ------------------------------------------
        n_cells = (battery_capacity * 1000) / (cell.Qnom * cell.Unom)  # Number of Cells // For scaling to single cell

    def calc_aging_naumann(self):

        # Pre-Calculation
        T_Cell = T_Cell + 273.15

        #  calendric stress factors
        k_temp_Q_cal = np.mean(1.2571e-05 * np.exp((-17126 / 8.3145) * (1 / T_Cell - 1 / 298.15)))
        k_temp_R_cal = np.mean(3.419e-10 * np.exp((-71827 / 8.3145) * (1 / T_Cell - 1 / 298.15)))
        k_soc_Q_cal = np.mean(2.85750 * ((SOC - 0.5) ** 3) + 0.60225)
        k_soc_R_cal = np.mean(3.3903 * ((SOC - 0.5) ** 2) + 1.56040)

        #  DOD calculation for cyclic aging
        dod, soc_mean = rfcounting(SOC)
        fec = sum(DODs)
        DODs = np.asarray(DODs)
        k_DOD_Q_cyc = sum((4.0253 * ((DODs - 0.6) ** 3) + 1.09230) * DODs) / fec
        k_DOD_R_cyc = sum((6.8477 * ((DODs - 0.5) ** 3) + 0.91882) * DODs) / fec

        Caging = abs(Crate)
        k_C_Q_cyc = sum(0.0971 + 0.063 * Caging) / len(Caging)
        k_C_R_cyc = sum(0.0023 - 0.0018 * Caging) / len(Caging)

        # Time in Seconds!
        t_aging = len(SOC) * dt  # Time for which aging is evaluated

        t_eq = (Q_loss_cal_prev / k_soc_Q_cal / k_temp_Q_cal) ** 2
        Qloss_new_cal = k_temp_Q_cal * k_soc_Q_cal * np.sqrt(t_eq + t_aging)
        Rinc_new_cal = R_inc_cal_prev + k_temp_R_cal * k_soc_R_cal * t_aging

        FEC_eq = (100 * Q_loss_cyc_prev / k_DOD_Q_cyc / k_C_Q_cyc) ** 2
        Qloss_new_cyc = 0.01 * (k_DOD_Q_cyc * k_C_Q_cyc) * np.sqrt(FEC_eq + fec)
        Rinc_new_cyc = R_inc_cyc_prev + 0.01 * (k_DOD_R_cyc * k_C_R_cyc) * fec

        Qloss_ges = Qloss_new_cal + Qloss_new_cyc
        Rinc_ges = Rinc_new_cal + Rinc_new_cyc

        return [Qloss_ges, Rinc_ges, Qloss_new_cal, Rinc_new_cal, Qloss_new_cyc, Rinc_new_cyc]

    def calc_aging_schmalstieg(cell, T_Cell, SOC, Crate, Q_loss_cal_prev, Q_loss_cyc_prev, R_inc_cyc_prev,
                               R_inc_cal_prev, dt):

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
        Qloss_new_cyc = k_VW * (k_DOD_Q_cyc + k_Uavgs_Q_cyc) * np.sqrt(
            Q_tot_eq + Q_tot_age)  # Scaled according to Teichert!
        Rinc_new_cyc = k_VW * (k_DOD_R_cyc + k_Uavgs_R_cyc) * (Q_tot_age)

        # Time = Days
        t_aging = len(SOC) * dt / 3600 / 24  # Time for which aging is evaluated
        t_eq_Q = (Q_loss_cal_prev / k_VW / k_temp_Q_cal / k_soc_Q_cal) ** (4 / 3)
        t_eq_R = (R_inc_cal_prev / k_VW / k_temp_R_cal / k_soc_R_cal) ** (4 / 3)

        Qloss_new_cal = k_VW * k_temp_Q_cal * k_soc_Q_cal * ((t_eq_Q + t_aging) ** 0.75)
        Rinc_new_cal = k_VW * k_temp_R_cal * k_soc_R_cal * ((t_eq_R + t_aging) ** 0.75)

        Qloss_ges = Qloss_new_cal + Qloss_new_cyc
        Rinc_ges = Rinc_new_cal + Rinc_new_cyc

        return [Qloss_ges, Rinc_ges, Qloss_new_cal, Rinc_new_cal, Qloss_new_cyc, Rinc_new_cyc]

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

    def rint_model(self, p_out):

        ocv = self.ocv_func(self.soc)
        if p_out > 0:
            r_i = self.r_i_ch_func(self.t_cell[-1], self.soc[-1])
        else:
            r_i_dch = self.r_i_dch_func(self.t_cell[-1], self.soc[-1])

        i = np.real((-ocv + np.sqrt((ocv ** 2) + (4 * r_i * p_out))) / (2 * r_i))
        p_loss = (i ** 2) * r_i

        return i, p_loss

    def thermal_model(bet, cell, n_cells, T_Cell, T_Housing, T_amb, P_Loss, Pcool, Pheat, dt):

        # Mass Calculation based on battery size
        m_battery_wo_cells = ((n_cells * cell.Qnom * cell.Unom) / cell.grav_energy_density) * (1 - cell.c2p_grav)
        Cth_Battery = cell.mass * 1045  # J/(kgK) OlafDiss

        Cth_Housing = bet.c_aluminium * m_battery_wo_cells  # J/K

        T_Housing_new = T_Housing + dt * (
                    ((Pcool * bet.COPcool) + (Pheat * bet.COPheat) + bet.k_bh * n_cells * (T_Cell -
                                                                                           T_Housing) + bet.k_out * (
                                 T_amb - T_Housing)) / Cth_Housing)

        T_Cell_new = T_Cell + dt * ((P_Loss + bet.k_bh * (T_Housing - T_Cell)) / Cth_Battery)

        return T_Cell_new, T_Housing_new

    def thermal_control(bet, T_Cell, p_cool_prev, p_value_control, p_value, n_cells):

        # Control Algorithm for active Cooling
        if T_Cell < bet.T_Heat:
            P_Heat = bet.Pheater
            P_Cool = 0
        elif T_Cell > bet.T_Cool_on or (
                T_Cell > bet.T_Cool_off and p_cool_prev > 0):  # Cool if Cooling-Threshold is exeeded or (if Cooling was active in the previous time step and Off-Threshold is not reached yet)
            P_Heat = 0
            P_Cool = bet.Pcooler
        else:
            P_Heat = 0
            P_Cool = 0

        # Impact of Cooling on Power

        # Case Driving // Cooling Power added to Power demand of driving task
        if p_value <= 0:
            p_value_control = p_value_control + (P_Cool + P_Heat) / n_cells

        # Case Charging // Cooling Power from Infrastructure -> If Cell is limiting no further power demand from cooling
        else:
            if p_value >= p_value_control:
                p_value_control = p_value_control - (P_Cool + P_Heat) / n_cells

        return p_value_control, P_Cool, P_Heat