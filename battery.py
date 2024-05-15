#!/usr/bin/env python3
"""
battery.py

--- Description ---
This script implements the Battery Aging Model for the degradation simulation in the mg_ev toolset.

For further information, see readme

--- Created by ---
Philipp Rosner

--- File Information ---
coding:     utf-8
license:    GPLv3
"""

###############################################################################
# Module imports
###############################################################################

import numpy as np
import os
import pickle
import pandas as pd
import rainflow
import scipy.interpolate as spip

import blocks

###############################################################################
# Class definitions
###############################################################################

class BatteryPackModel:

    def __init__(self, scenario, commodity):

        self.parent = commodity

        if isinstance(commodity, blocks.MobileCommodity):
            self.opt = self.parent.parent.opt
            self.chemistry = self.parent.parent.chemistry.lower()  # 'lfp' or 'nmc'
        else:  # StationaryEnergyStorage
            self.opt = self.parent.opt
            self.chemistry = self.parent.chemistry.lower()

        # Thermal model parameters
        self.c_th_spec_housing = 896  # Specific heat capacity of the pack housing (made from Al) in J/(kg K)
        self.c_th_spec_cell = 1045  # Specific heat capacity of LI cells as per Teichert's dissertation in J/(kg K)
        self.k_c2h = 0.899  # Thermal conductance between cell and housing as per Teichert's dissertation in W/K
        self.k_h2a = 10.9  # Thermal conductance between housing and ambient as per Teichert's dissertation in W/K

        # Active thermal control system parameters
        # self.p_cool = 10e3  # System cooling power in W [Schimpe et al.]
        # self.p_heat = 11.2e3  # System heating power in W [Schimpe et al.]
        # self.cop_cool = -3  # Coefficient of performance of cooling system in pu [Schimpe et al.]
        # self.cop_heat = 4  # Coefficient of performance of heating system in pu [Schimpe et al., Danish Energy Agency 2012]
        # self.t_cool_on = 35  # Cooling activation threshold in °C
        # self.t_cool_off = 30  # Cooling deactivation threshold in °C
        # self.t_heat_on = 10  # Heating activation threshold in °C
        # self.t_heat_off = 15 # Heating deactivation threshold in °C

        # Initial values for aging state tracking
        self.q_loss_cal = np.zeros(scenario.nhorizons)
        self.r_inc_cal = np.zeros(scenario.nhorizons)
        self.q_loss_cyc = np.zeros(scenario.nhorizons)
        self.r_inc_cyc = np.zeros(scenario.nhorizons)

        # Placeholders for aging variables to be filled every aging evaluation
        self.soc_hor = self.ocv_hor = self.cycles_hor = None
        self.temp_hor_c = self.temp_hor_k = self.p_hor = self.t_hor = None

        # Placeholders for pack level variables to be filled after component sizing in first horizon
        self.size = self.n_cells = self.m_cells = self.m_housing = self.c_th_cells = self.c_th_housing = None

        if self.chemistry == 'nmc':
            # Cell from Schmalstieg et al. - Sanyo UR18650E
            self.q_nom_cell = 2.15  # Typical capacity in Ah
            self.u_nom_cell = 3.6  # Nominal voltage in V
            self.u_min_cell = 3.0  # Minimum voltage in V
            self.u_max_cell = 4.2  # Maximum voltage in V
            self.i_max_cont_cell = 2.05  # Maximum charging current in A
            self.i_min_cont_cell = -6.15  # Maximum discharging current in A
            self.m_cell = 0.0445  # Cell mass in kg
            self.v_cell = 0.0165  #Cell volume in L

            self.e_spec_grav_c2p = 0.59  # Transformation factor of gravimetric energy density from cell to pack level
            self.e_spec_vol_c2p = 0.39  # Transformation factor of volumetric energy density from cell to pack level

            self.data_path = os.path.join(os.getcwd(), 'input', 'battery', 'sanyo_ur18650e.pkl')

        elif self.chemistry == 'lfp':
            # Cell from Naumann et al. - Sony US26650
            self.q_nom_cell = 3.0  # Typical capacity in Ah
            self.u_nom_cell = 3.2  # Nominal voltage in V
            self.u_min_cell = 2.0  # Minimum voltage in V
            self.u_max_cell = 3.6  # Maximum voltage in V
            self.i_max_cont_cell = 3  # Maximum charging current in A
            self.i_min_cont_cell = -20  # Maximum discharging current in A
            self.m_cell = 0.0845  # Cell mass in kg
            self.v_cell = 0.0345  #Cell volume in L

            self.e_spec_grav_c2p = 0.71  # Transformation factor of gravimetric energy density from cell to pack level
            self.e_spec_vol_c2p = 0.55  # Transformation factor of volumetric energy density from cell to pack level

            self.data_path = os.path.join(os.getcwd(), 'input', 'battery', 'sony_us26650.pkl')

        with open(self.data_path, 'rb') as file:
            self.ocv, self.r_i_ch, self.r_i_dch = pickle.load(file)

        self.ocv_interp = spip.RegularGridInterpolator(points=(self.ocv.index.to_numpy(),),
                                                       values=self.ocv.to_numpy(),
                                                       method='linear',
                                                       bounds_error=False,
                                                       fill_value=None)
        self.r_i_ch_interp = spip.RegularGridInterpolator(points=(self.r_i_ch.index.to_list(),
                                                                  self.r_i_ch.columns.to_list()),
                                                          values=self.r_i_ch.to_numpy(),
                                                          method='linear',
                                                          bounds_error=False,
                                                          fill_value=None)
        self.r_i_dch_interp = spip.RegularGridInterpolator(points=(self.r_i_dch.index.to_list(),
                                                                   self.r_i_dch.columns.to_list()),
                                                           values=self.r_i_dch.to_numpy(),
                                                           method='linear',
                                                           bounds_error=False,
                                                           fill_value=None)

        self.e_cell = self.q_nom_cell * self.u_nom_cell  # Nominal energy content of the cell in Wh
        self.e_spec_grav_cell = self.e_cell / self.m_cell
        self.e_spec_vol_cell = self.e_cell / self.v_cell
        self.c_th_cell = self.m_cell * self.c_th_spec_cell

    def age(self, commodity, run, scenario, horizon):
        """
        Get aging relevant features for control horizon, apply correct aging model,
        and derate block for next horizon
        """

        if horizon.index == 0:  # first horizon of simulation - pack level values dependent on size are not set yet
            self.get_pack_parameters()

        # Calculate power requirement and C-rate on cell level
        # Charge power is positive, discharging power is negative
        if isinstance(commodity, blocks.MobileCommodity):
            self.p_cell_hor = (commodity.flow_bat_in_ch - commodity.flow_bat_out_ch) / self.n_cells
        else:  # StationaryStorage
            self.p_cell_hor = (commodity.flow_in_ch - commodity.flow_out_ch) / self.n_cells
        self.crate_hor = self.p_cell_hor / self.e_cell

        # Get SOC & OCV timeseries from horizon results
        self.soc_hor = commodity.soc_ch[:-1]  # omit last step as its effects reach into next horizon
        self.ocv_hor = pd.DataFrame(data=self.ocv_interp(self.soc_hor), index=self.soc_hor.index).squeeze()

        # Calculate timespan of horizon in seconds
        self.t_hor = (self.soc_hor.index[-1] - self.soc_hor.index[0]).total_seconds()

        # Get temperature timeseries
        if isinstance(commodity.temp_battery, str):
            try:
                self.temp_hor_c = scenario.blocks[commodity.temp_battery].data_ph.loc[horizon.dti_ch, 'temp_air']
            except KeyError or NameError:
                run.logger.warning(f'Scenario {scenario.name}: Battery temp source for storage {self.parent.name}'
                                   f' not found, using 25°C default')
                self.temp_hor_c = pd.Series(data=25, index=horizon.dti_ch)
        elif isinstance(commodity.temp_battery, (int, float)):
            self.temp_hor_c = pd.Series(data=commodity.temp_battery, index=horizon.dti_ch)  # pack temperature in °C
        else:
           ValueError('Battery temperature must be the name of a PVSource block or numeric')

        self.temp_hor_k = self.temp_hor_c + 273.15  # temperature conversion to Kelvin

        # Determine DODs and mean SOCs of (half) cycles within the horizon using the ASTM E 1049-85 norm
        self.cycles_hor = {'depth': [], 'mean': [], 'type': []}
        for range, mean, count, _, _ in rainflow.extract_cycles(self.soc_hor):
            self.cycles_hor['depth'].append(range)  # depth of cycle expressed as SOC fraction
            self.cycles_hor['mean'].append(mean)  # mean SOC of cycle
            self.cycles_hor['type'].append(count)  # type of cycle 0.5 (half cycle) or 1 (full cycle)
        self.cycles_hor['depth'] = np.array(self.cycles_hor['depth'])
        self.cycles_hor['mean'] = np.array(self.cycles_hor['mean'])
        self.cycles_hor['type'] = np.array(self.cycles_hor['type'])

        # Calculate Number of Full Equivalent Cycles (1 EFC is 2 capacities of charge throughput)
        self.fec_hor = sum(self.cycles_hor['type'] * self.cycles_hor['depth'])
        self.q_tot_hor = self.fec_hor * (2 * self.q_nom_cell)

        # Determine actual aging
        if self.chemistry == 'nmc':
            self.calc_aging_schmalstieg(horizon)
        elif self.chemistry == 'lfp':
            self.calc_aging_naumann(horizon)

        # Update block / commodity storage size
        commodity.soh = 1 - sum(self.q_loss_cyc) - sum(self.q_loss_cal)
        commodity.soc_min = (1 - commodity.soh) / 2
        commodity.soc_max = 1 - ((1 - commodity.soh) / 2)

    def calc_aging_naumann(self, horizon):

        # Set global tuning factor
        k_tuning = 1

        #  Calculate calendric stress factor timeseries (from https://doi.org/10.1016/j.est.2018.01.019)
        k_temp_q_cal = 1.2571e-05 * np.exp((-17126 / 8.3145) * (1 / self.temp_hor_k - 1 / 298.15))
        k_temp_r_cal = 3.419e-10 * np.exp((-71827 / 8.3145) * (1 / self.temp_hor_k - 1 / 298.15))
        k_soc_q_cal = 2.85750 * ((self.soc_hor - 0.5) ** 3) + 0.60225
        k_soc_r_cal = 3.3903 * ((self.soc_hor - 0.5) ** 2) + 1.56040

        # Aggregate calendric stress factors (converting them to scalar)
        k_temp_q_cal = k_temp_q_cal.mean()
        k_temp_r_cal = k_temp_r_cal.mean()
        k_soc_q_cal = k_soc_q_cal.mean()
        k_soc_r_cal = k_soc_r_cal.mean()

        # Calculate previous aging state as equivalent time at current conditions
        t_eq = (np.sum(self.q_loss_cal) / ((k_tuning * k_soc_q_cal * k_temp_q_cal) ** 2))

        # Calculate calendric aging within this horizon
        self.q_loss_cal[horizon.index] = (k_tuning *
                                          k_temp_q_cal *
                                          k_soc_q_cal *
                                          (np.sqrt(t_eq + self.t_hor) - np.sqrt(t_eq)))
        self.r_inc_cal[horizon.index] = (k_tuning *
                                         k_temp_r_cal *
                                         k_soc_r_cal *
                                         self.t_hor)  # linear - no equivalent time needed

        # Calculate cyclic stress factor series (DOD for each detected cycle, C-rate over time)
        # Methodology from https://doi.org/10.1016/j.jpowsour.2019.227666
        k_dod_q_cyc = 4.0253 * ((self.cycles_hor['depth'] - 0.6) ** 3) + 1.09230
        k_dod_r_cyc = 6.8477 * ((self.cycles_hor['depth'] - 0.5) ** 3) + 0.91882
        k_crate_q_cyc = 0.0971 + 0.063 * self.crate_hor
        k_crate_r_cyc = 0.0023 - 0.0018 * self.crate_hor

        if np.sum(self.cycles_hor['depth']) > 0:  # actual cycling happened
            # Aggregate DOD stress factors through DOD-weighted mean (converting them to scalar)
            k_dod_q_cyc = np.sum(k_dod_q_cyc * self.cycles_hor['depth']) / np.sum(self.cycles_hor['depth'])
            k_dod_r_cyc = np.sum(k_dod_r_cyc * self.cycles_hor['depth']) / np.sum(self.cycles_hor['depth'])

            # Aggregate C-rate stress factors through arithmetic mean (converting them to a scalar)
            k_crate_q_cyc = k_crate_q_cyc.mean()
            k_crate_r_cyc = k_crate_r_cyc.mean()

            # Define previous aging state as equivalent FECs at current conditions
            fec_eq = 100 * np.sum(self.q_loss_cyc) / ((k_tuning * k_dod_q_cyc * k_crate_q_cyc) ** 2)

            # Calculate cyclic aging within this horizon (0.01 converts percent to fraction)
            self.q_loss_cyc[horizon.index] = (0.01 *
                                              (k_tuning * k_dod_q_cyc * k_crate_q_cyc) *
                                              (np.sqrt(fec_eq + self.fec_hor) - np.sqrt(fec_eq)))
            self.r_inc_cyc[horizon.index] = (0.01 *
                                             (k_tuning * k_dod_r_cyc * k_crate_r_cyc) *
                                             self.fec_hor)  # linear, not fec_eq needed
        else:  # technically not necessary to set values here as no aging happens, eases debugging
            beta_cap = 0
            beta_res = 0
            q_eq = 0

    def calc_aging_schmalstieg(self, horizon):

        # Schmalstieg aging model is not verified yet against aging data from original paper

        # Set global tuning factor
        # k_tuning = 0.43  # Teichert for VW ID.3 cell
        k_tuning = 1  # deactivation of tuning factor

        #  Calculate calendric stress factor timeseries (from http://dx.doi.org/10.1016/j.jpowsour.2014.02.012)
        alpha_cap = (7.543 * self.ocv_hor - 23.75) * 1e6 * np.exp(-6976 / self.temp_hor_k)  # timeseries over all steps
        alpha_res = (5.270 * self.ocv_hor - 16.32) * 1e5 * np.exp(-5986 / self.temp_hor_k)  # timeseries over all steps

        # Aggregate calendric stress factors (converting them to scalar) and limit them to zero to avoid
        # a) negative aging and b) problems in calculation of t_eq
        alpha_cap = np.maximum(alpha_cap.mean(), 1E-10)
        alpha_res = np.maximum(alpha_res.mean(), 1E-10)

        # Calculate previous aging state as equivalent time at current conditions
        t_eq_q = (np.sum(self.q_loss_cal) / (k_tuning * alpha_cap)) ** (4 / 3)
        t_eq_r = (np.sum(self.r_inc_cal) / (k_tuning * alpha_res)) ** (4 / 3)

        # Calculate calendric aging in this horizon
        t_hor_days = self.t_hor / (3600 * 24)  # Schmalstieg model is evaluated in days
        self.q_loss_cal[horizon.index] = k_tuning * alpha_cap * ((t_eq_q + t_hor_days) ** 0.75 - t_eq_q ** 0.75)
        self.r_inc_cal[horizon.index] = k_tuning * alpha_cap * ((t_eq_r + t_hor_days) ** 0.75 - t_eq_r ** 0.75)

        # Calculate mean OCV of each detected cycle
        ocv_cycles_mean = self.ocv_interp(self.cycles_hor['mean']).reshape([-1,])
        # TODO Schmalstieg states quadratic mean (rms) of voltage instead of arithmetic mean!

        # Calculate cyclic stress factor series for each cycle
        beta_cap = 7.348E-3 * (ocv_cycles_mean - 3.667) ** 2 + 7.6E-4 + 4.081E-3 * self.cycles_hor['depth']
        beta_res = 2.153E-4 * (ocv_cycles_mean - 3.725) ** 2 - 1.521E-5 + 2.798E-4 * self.cycles_hor['depth']
        beta_res = np.maximum(1.5E-5, beta_res)  # limitation as per text following Eq. (21) in paper

        if np.sum(self.cycles_hor['depth']) > 0:  # actual cycling happened

            # Aggregate cyclic stress factors through DOD-weighted mean (converting them to scalar)
            beta_cap = np.sum(beta_cap * self.cycles_hor['depth']) / np.sum(self.cycles_hor['depth'])
            beta_res = np.sum(beta_res * self.cycles_hor['depth']) / np.sum(self.cycles_hor['depth'])

            # Define previous aging state as equivalent FECs at current conditions
            q_eq = (sum(self.q_loss_cyc) / (k_tuning * beta_cap)) ** 2

            # Calculate cyclic aging
            self.q_loss_cyc[horizon.index] = k_tuning * beta_cap * (np.sqrt(q_eq + self.q_tot_hor) - np.sqrt(q_eq))
            self.r_inc_cyc[horizon.index] = k_tuning * beta_res * self.q_tot_hor

        else:  # technically not necessary to set values here as no aging happens, eases debugging
            beta_cap = 0
            beta_res = 0
            q_eq = 0

    def get_pack_parameters(self):
        self.size = self.parent.size
        # Calculate number of cells as a float to correctly represent power split with nonreal cells
        self.n_cells = self.size / self.e_cell
        self.m_cells = self.n_cells * self.m_cell
        self.m_housing = self.m_cells * (1 - self.e_spec_grav_c2p)
        self.c_th_cells = self.n_cells * self.c_th_cell
        self.c_th_housing = self.c_th_spec_housing * self.m_housing

    def rint_model(self, p_out):
        """
        This function calculates output voltage and current of a battery cell based on a simple Zero-RC Equivalent
        Circuit Model consisting of an ideal voltage source and a series resistance. Power loss to heat at the series
        resistor can also be evaluated
        """

        ocv = self.ocv_func(self.soc)

        # Charge power is positive, Discharge is negative
        if p_out > 0:
            r_i = self.r_i_ch_func(self.t_cell[-1], self.soc[-1])
        else:  # Case p_out = 0 is irrelevant, as current is 0 anyway
            r_i = self.r_i_dch_func(self.t_cell[-1], self.soc[-1])

        i = np.real((-ocv + np.sqrt((ocv ** 2) + (4 * r_i * p_out))) / (2 * r_i))
        p_loss = r_i * (i ** 2)

        return i, p_loss

    # def thermal_model(self):
    #
    #     temp_housing_new = temp_housing + dt * (
    #                 ((Pcool * bet.COPcool) + (Pheat * bet.COPheat) + bet.k_bh * n_cells * (T_Cell -
    #                                                                                        T_Housing) + bet.k_out * (
    #                              T_amb - T_Housing)) / Cth_Housing)
    #
    #     T_Cell_new = T_Cell + dt * ((P_Loss + bet.k_bh * (T_Housing - T_Cell)) / Cth_Battery)
    #
    #     return T_Cell_new, T_Housing_new

    # def thermal_control(bet, T_Cell, p_cool_prev, p_value_control, p_value, n_cells):
    #
    #     # Control Algorithm for active Cooling
    #     if T_Cell < bet.T_Heat:
    #         P_Heat = bet.Pheater
    #         P_Cool = 0
    #     elif T_Cell > bet.T_Cool_on or (
    #             T_Cell > bet.T_Cool_off and p_cool_prev > 0):  # Cool if Cooling-Threshold is exeeded or (if Cooling was active in the previous time step and Off-Threshold is not reached yet)
    #         P_Heat = 0
    #         P_Cool = bet.Pcooler
    #     else:
    #         P_Heat = 0
    #         P_Cool = 0
    #
    #     # Impact of Cooling on Power
    #
    #     # Case Driving // Cooling Power added to Power demand of driving task
    #     if p_value <= 0:
    #         p_value_control = p_value_control + (P_Cool + P_Heat) / n_cells
    #
    #     # Case Charging // Cooling Power from Infrastructure -> If Cell is limiting no further power demand from cooling
    #     else:
    #         if p_value >= p_value_control:
    #             p_value_control = p_value_control - (P_Cool + P_Heat) / n_cells
    #
    #     return p_value_control, P_Cool, P_Heat