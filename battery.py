import numpy as np
import os
import pickle
import rainflow
import scipy.interpolate as spip

from dateutil.relativedelta import relativedelta

class BatteryPack:

    def __init__(self, scenario, commodity):

        self.parent = commodity
        self.chemistry = self.parent.parent.chemistry.lower()  # 'lfp' or 'nmc'

        self.soh = 1  # initial value

        self.soc_max = 1  # Upper SOC limit
        self.soc_min = 0  # Lower SOC limit

        self.c_cell = 76.27  # Cell heat capacity as per Forgez et al. # todo find unit
        self.c_aluminium = 896  # Specific heat capacity of the pack housing (made from Al) in J/(kg K)
        self.r_th_ch = 3.3  # Thermal resistance between cell and pack housing # todo find unit
        self.r_th_ha = 4.343  # Thermal resistance between pack housing and ambient in W/K [Neubauer et al.]
        # self.k_bh = 0.899  # K/W #OlafDiss
        # self.k_out = 10.9  # W/K OlafDiss

        # self.p_cool = 10e3  # System cooling power in W [Schimpe et al.]
        # self.p_heat = 11.2e3  # System heating power in W [Schimpe et al.]
        # self.cop_cool = -3  # Coefficient of performance of cooling system in pu [Schimpe et al.]
        # self.cop_heat = 4  # Coefficient of performance of heating system in pu [Schimpe et al., Danish Energy Agency 2012]
        # self.t_cool_on = 35  # Cooling activation threshold in °C
        # self.t_cool_off = 30  # Cooling deactivation threshold in °C
        # self.t_heat_on = 10  # Heating activation threshold in °C
        # self.t_heat_off = 15 # Heating deactivation threshold in °C

        # self.e_spec_grav_c2p = [0.59, 0.71]
        # self.e_spec_vol_c2p = [0.39, 0.55]

        # Initial values for aging state tracking
        self.q_loss_cal = np.zeros(scenario.nhorizons)
        self.r_inc_cal = np.zeros(scenario.nhorizons)
        self.q_loss_cyc = np.zeros(scenario.nhorizons)
        self.r_inc_cyc = np.zeros(scenario.nhorizons)

        # Placeholders for aging variables to be filled every aging evaluation
        self.soc_hor = self.cycles_hor = self.temp_hor_c = self.temp_hor_k = self.p_hor = None

        if self.chemistry == 'nmc':
            # Cell from Schmalstieg et al. - Sanyo UR18650E
            self.q_nom_cell = 2.15  # Typical capacity in Ah
            self.u_nom = 3.6  # Nominal voltage in V
            self.u_min = 3.0  # Minimum voltage in V
            self.u_max = 4.2  # Maximum voltage in V
            self.i_max_cont = 2.05  # Maximum charging current in A
            self.i_min_cont = -6.15  # Maximum discharging current in A
            self.m = 0.0445  # Cell mass in kg
            self.v = 0.0165  #Cell volume in L
            self.data_path = os.path.join(os.getcwd(), 'input', 'battery', 'sanyo_ur18650e.pkl')

        elif self.chemistry == 'lfp':
            # Cell from Naumann et al. - Sony US26650
            self.q_nom_cell = 3.0  # Typical capacity in Ah
            self.u_nom = 3.2  # Nominal voltage in V
            self.u_min = 2.0  # Minimum voltage in V
            self.u_max = 3.6  # Maximum voltage in V
            self.i_max_cont = 3  # Maximum charging current in A
            self.i_min_cont = -20  # Maximum discharging current in A
            self.m_cell = 0.0845  # Cell mass in kg
            self.v_cell = 0.0345  #Cell volume in L
            self.data_path = os.path.join(os.getcwd(), 'input', 'battery', 'sony_us26650.pkl')

        with open(self.data_path, 'rb') as file:
            self.ocv, self.r_i_ch, self.r_i_dch = pickle.load(file)

        self.ocv_interp = spip.RegularGridInterpolator(points=(self.ocv.index.to_numpy(),),
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

        self.e_cell = self.q_nom_cell * self.u_nom  # Nominal energy content of the cell in Wh
        self.e_spec_grav_cell = self.e_cell / self.m_cell
        self.e_spec_vol_cell = self.e_cell / self.v_cell

        if self.parent.parent.opt:  # self.parent.parent is CommoditySystem
            self.size = None  # placeholder for nominal pack capacity - to be filled after size optimization
        else:
            self.size = self.parent.size
            # Calculate number of cells as a float to correctly represent power split with nonreal cells
            self.n_cells = self.size / self.e_cell

    def age(self, block, horizon):
        """
        Get aging relevant features for control horizon, apply correct aging model,
        and derate block for next horizon
        """
        # Calculate timespan of horizon in seconds
        self.t_hor = (self.soc_hor.index[-1] - self.soc_hor.index[0]).total_seconds()

        # Calculate power requirement and C-rate on cell level
        # Charge power is positive, discharging power is negative
        self.p_cell_hor = (block.flow_bat_in_ch - block.flow_bat_out_ch) / self.n_cells
        self.crate_hor = self.p_cell_hor / self.e_cell

        # Get SOC & OCV timeseries from horizon results
        self.soc_hor = block.soc_ch
        self.ocv_hor = self.ocv_interp(self.soc_hor)

        # Get temperature timeseries
        self.temp_hor_c = 25  # pack temperature in °C # todo replace with ambient (or thermal model)
        self.temp_hor_k = self.temp_hor_c + 273.15  # temperature conversion to Kelvin

        # Determine DODs and mean SOCs of (half) cycles within the horizon using the ASTM E 1049-85 norm
        self.cycles_hor = [{'range': rng, 'mean': mean, 'count': count}
                           for rng, mean, count, _, _
                           in rainflow.extract_cycles(self.soc_hor)]

        # Calculate Number of Full Equivalent Cycles (1 EFC is 2 capacities of charge throughput)
        self.fec_hor = sum([cycle['count'] * cycle['range'] for cycle in self.cycles_hor])
        self.q_tot_hor = 2 * self.fec_hor * self.q_nom_cell   # todo check accuracy - possibly remove factor

        # Calculate Depths of Discharge
        self.dod_hor = np.array([cycle['range'] for cycle in self.cycles_hor])  # todo how to count half cycles ?

        if self.chemistry == 'nmc':
            self.calc_aging_schmalstieg(horizon)
        elif self.chemistry == 'lfp':
            self.calc_aging_naumann(horizon)

        # update block's storage size
        block.soh = 1 - sum(self.q_loss_cyc) - sum(self.q_loss_cal)

    def calc_aging_naumann(self, horizon):

        #  Calculate calendric stress factor timeseries (from https://doi.org/10.1016/j.est.2018.01.019)
        k_temp_q_cal = 1.2571e-05 * np.exp((-17126 / 8.3145) * (1 / self.temp_hor_k - 1 / 298.15))
        k_temp_r_cal = 3.419e-10 * np.exp((-71827 / 8.3145) * (1 / self.temp_hor_k - 1 / 298.15))
        k_soc_q_cal = 2.85750 * ((self.soc_hor - 0.5) ** 3) + 0.60225
        k_soc_r_cal = 3.3903 * ((self.soc_hor - 0.5) ** 2) + 1.56040

        # Aggregate calendric stress factors (converting them to scalar)
        for stress_factor in [k_temp_q_cal, k_temp_r_cal, k_soc_q_cal, k_soc_r_cal]:
            stress_factor = np.mean(stress_factor)

        # Calculate previous aging state as equivalent time at current conditions
        t_eq = (np.sum(self.q_loss_cal) / k_soc_q_cal / k_temp_q_cal) ** 2

        # Calculate calendric aging
        self.q_loss_cal[horizon.index] = k_temp_q_cal * k_soc_q_cal * np.sqrt(t_eq + self.t_hor)
        self.r_inc_cal[horizon.index] = k_temp_r_cal * k_soc_r_cal * self.t_hor  # linear - no equivalent time needed

        # Calculate cyclic stress factor series (DOD for each detected cycle, C-rate over time)
        # Methodology from https://doi.org/10.1016/j.jpowsour.2019.227666
        k_dod_q_cyc = 4.0253 * ((self.dod_hor - 0.6) ** 3) + 1.09230
        k_dod_r_cyc = 6.8477 * ((self.dod_hor - 0.5) ** 3) + 0.91882
        k_crate_q_cyc = 0.0971 + 0.063 * self.crate_hor
        k_crate_r_cyc = 0.0023 - 0.0018 * self.crate_hor

        # Aggregate DOD stress factors through DOD-weighted mean (converting them to scalar)
        for stress_factor in [k_dod_q_cyc, k_dod_r_cyc]:
            stress_factor = np.sum(stress_factor * self.dod_hor) / np.sum(self.dod_hor)

        # Aggregate C-rate stress factors through arithmetic mean (converting them to a scalar)
        for stress_factor in [k_crate_q_cyc, k_crate_r_cyc]:
            stress_factor = np.mean(stress_factor)

        # Define previous aging state as equivalent FECs at current conditions
        fec_eq = (100 * np.sum(self.q_loss_cyc) / k_dod_q_cyc / k_crate_q_cyc) ** 2

        # Calculate cyclic aging (0.01 converts percent to fraction)
        self.q_loss_cyc[horizon.index] = 0.01 * (k_dod_q_cyc * k_crate_q_cyc) * np.sqrt(fec_eq + self.fec_hor)
        self.r_inc_cyc[horizon.index] = 0.01 * (k_dod_r_cyc * k_crate_r_cyc) * self.fec_hor # linear, not fec_eq needed

    def calc_aging_schmalstieg(self, horizon):

        # Set global tuning factor
        # k_tuning = 0.43  # Teichert for VW ID.3 cell
        k_tuning = 1  # deactivation of tuning factor

        #  Calculate calendric stress factor timeseries (from http://dx.doi.org/10.1016/j.jpowsour.2014.02.012)
        alpha_cap = (7.543 * self.ocv_hor - 23.75) * 1e6 * np.exp(-6976 / self.temp_hor_k)  # timeseries over all steps
        alpha_res = (5.270 * self.ocv_hor - 16.32) * 1e5 * np.exp(-5986 / self.temp_hor_k)  # timeseries over all steps

        # Aggregate calendric stress factors (converting them to scalar)
        for stress_factor in [alpha_cap, alpha_res]:
            stress_factor = np.mean(stress_factor)

        # Calculate previous aging state as equivalent time at current conditions
        t_eq_Q = (np.sum(self.q_loss_cal) / (k_tuning * alpha_cap)) ** (4 / 3)
        t_eq_R = (np.sum(self.r_inc_cal) / (k_tuning * alpha_res)) ** (4 / 3)

        # Calculate calendric aging
        self.q_loss_cal[horizon.index] = k_tuning * alpha_cap * ((t_eq_Q + t_aging) ** 0.75)
        self.r_inc_cal[horizon.index] = k_tuning * alpha_cap * ((t_eq_R + t_aging) ** 0.75)

        # Calculate mean OCV of each detected cycle
        ocv_cycles_mean = self.ocv_interp(self.cycles_hor['mean'])
        # TODO Schmalstieg states quadratic mean (rms) of voltage instead of arithmetic mean!

        # Calculate cyclic stress factor series for each cycle
        beta_cap = 7.348E-3 * (ocv_cycles_mean - 3.667) ** 2 + 7.6E-4 + 4.081E-3 * self.dod_hor
        beta_res = 2.153E-4 * (ocv_cycles_mean - 3.725) ** 2 + 1.521E-5 + 2.798E-4 * self.dod_hor

        # Aggregate cyclic stress factors through DOD-weighted mean (converting them to scalar)
        for stress_factor in [beta_cap, beta_res]:
            stress_factor = np.sum(stress_factor * self.dod_hor) / np.sum(self.dod_hor)

        # Q_tot_age = Qtot * 2  # is this conversion to FEC or sth?
        q_eq = sum(self.q_loss_cyc) / (k_tuning * (beta_cap ** 2))  # todo check against paper

        # Calculate cyclic aging
        self.q_loss_cyc[horizon.index] = k_tuning * beta_cap * ((Q_tot_eq + Q_tot_age) ** 0.5)
        self.r_inc_cal[horizon.index] = k_tuning * beta_res * Q_tot_age


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