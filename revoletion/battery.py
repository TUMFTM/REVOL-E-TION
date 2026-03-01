#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import rainflow
import scipy.interpolate as spip


@dataclass(frozen=True)
class BatteryCell:
    q_nom: float  # Typical capacity in Ah
    u_nom: float  # Nominal voltage in V
    u_min: float  # Minimum voltage in V
    i_max_cont: float  # Maximum charging current in A
    i_min_cont: float  # Maximum discharging current in A
    mass: float  # Cell mass in kg
    volume: float  # Cell volume in L

    # ToDo: move C2P parameters to pack level
    e_spec_grav_c2p: float  # Transformation factor of gravimetric energy density from cell to pack level
    e_spec_vol_c2p: float  # Transformation factor of volumetric energy density from cell to pack level

    @classmethod
    def from_json(cls, path_file: Path):
        with open(path_file, "r") as f:
            data = json.load(f)
        return cls(
            q_nom=data["q_nom"],
            u_nom=data["u_nom"],
            u_min=data["u_min"],
            i_max_cont=data["i_max_cont"],
            i_min_cont=data["i_min_cont"],
            mass=data["mass"],
            volume=data["volume"],
            e_spec_grav_c2p=data["e_spec_grav_c2p"],
            e_spec_vol_c2p=data["e_spec_vol_c2p"],
        )

    @cached_property
    def e(self) -> float:
        return self.q_nom * self.u_nom  # Nominal energy content of the cell in Wh

    @cached_property
    def e_spec_grav(self) -> float:
        return self.e / self.mass  # Gravimetric energy density of the cell in Wh/kg

    @cached_property
    def e_spec_vol(self) -> float:
        return self.e / self.volume  # Volumetric energy density of the cell in Wh/L

    @cached_property
    def c_th(self) -> float:
        raise NotImplementedError("Specific heat capacity is not implemented yet.")
        return self.m_cell * self.c_th_spec_cell  # Thermal capacity of the cell in J/K


class BatteryPackModel(ABC):
    _CELL: str

    def __init__(self, block):
        self.block = block
        self.scenario = self.block.scenario

        self.chemistry = self.block.chemistry

        # Thermal model parameters
        # self.c_th_spec_housing = 896  # Specific heat capacity of the pack housing (made from Al) in J/(kg K)
        # self.c_th_spec_cell = 1045  # Specific heat capacity of LI cells as per Teichert's dissertation in J/(kg K)
        # self.k_c2h = 0.899  # Thermal conductance between cell and housing as per Teichert's dissertation in W/K
        # self.k_h2a = 10.9  # Thermal conductance between housing and ambient as per Teichert's dissertation in W/K

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
        self.q_loss_cal = np.zeros(self.scenario.nhorizons + 1)
        self.r_inc_cal = np.zeros(self.scenario.nhorizons + 1)
        self.q_loss_cyc = np.zeros(self.scenario.nhorizons + 1)
        self.r_inc_cyc = np.zeros(self.scenario.nhorizons + 1)

        # set initial aging state. Neglected for r_inc_cal and r_inc_cyc as REVOL-E-TION doesn't take them into account
        # Horizon 0 is previous history before simulation --> initial horizon is 1 --> hor_battery = hor_sim + 1
        self.q_loss_cal[0] = self.block.states.loc[self.scenario.times.sim.start, "q_loss_cal"]
        self.q_loss_cyc[0] = self.block.states.loc[self.scenario.times.sim.start, "q_loss_cyc"]

        # Placeholders for pack level variables to be filled after component sizing in first horizon
        self.size = self.n_cells = self.m_cells = self.m_housing = self.c_th_cells = self.c_th_housing = None

        self.cell = BatteryCell.from_json(self.scenario.paths.data_persist / f"{self._CELL}_parameters.json")

        with open(self.scenario.paths.data_persist / f"{self._CELL}.pkl", "rb") as file:
            self.ocv, self.r_i_ch, self.r_i_dch = pickle.load(file)

        self.ocv_interp = spip.RegularGridInterpolator(
            points=(self.ocv.index.to_numpy(),),
            values=self.ocv.to_numpy(),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        self.r_i_ch_interp = spip.RegularGridInterpolator(
            points=(self.r_i_ch.index.to_list(), self.r_i_ch.columns.to_list()),
            values=self.r_i_ch.to_numpy(),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        self.r_i_dch_interp = spip.RegularGridInterpolator(
            points=(self.r_i_dch.index.to_list(), self.r_i_dch.columns.to_list()),
            values=self.r_i_dch.to_numpy(),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    @classmethod
    def from_block(cls, block) -> NMCBatteryPackModel | LFPBatteryPackModel:
        if block.chemistry == "nmc":
            return NMCBatteryPackModel(block)
        elif block.chemistry == "lfp":
            return LFPBatteryPackModel(block)
        else:
            raise ValueError(f"Unsupported battery chemistry {block.chemistry}")

    def age(self, horizon):
        """
        Get aging relevant features for control horizon, apply correct aging model,
        and derate block for next horizon
        """

        # If aging is disabled, keep initial SOH
        if not self.block.aging:
            self.block.states.loc[horizon.ch.end, "soh"] = self.block.states.loc[horizon.ph.start, "soh"]
            return

        if horizon.index == 0:  # first horizon of simulation - pack level values dependent on size are not set yet
            self.get_pack_parameters()

        # Calculate power requirement and C-rate on cell level
        # Charge power is positive, discharging power is negative
        p_cell_hor = (
            self.block.flows.loc[horizon.ch.dti, "bat_in"] - self.block.flows.loc[horizon.ch.dti, "bat_out"]
        ) / self.n_cells
        crate_hor = p_cell_hor / self.cell.e

        # Get SOC & OCV timeseries from horizon results
        soc_hor = self.block.states.loc[horizon.ch.dti_extd, "soc"]

        ocv_hor = pd.DataFrame(data=self.ocv_interp(soc_hor), index=soc_hor.index).squeeze()

        # Calculate timespan of horizon in seconds
        t_hor = (soc_hor.index[-1] - soc_hor.index[0]).total_seconds()

        # Get temperature timeseries
        if isinstance(self.block.temp_battery, str):
            try:
                temp_hor_c = self.scenario.block_registry.get("TopLevelBlock", {})[self.block.temp_battery].data.loc[
                    horizon.ch.dti, "temp_air"
                ]
            except KeyError or NameError:
                self.scenario.logger.warning(
                    f"Battery temp source for storage {self.block.name} not found - Using scenario temperature"
                )
                temp_hor_c = self.block.scenario.temp_air[horizon.ch.dti]
        elif isinstance(self.block.temp_battery, (int, float)):
            temp_hor_c = pd.Series(data=self.block.temp_battery, index=horizon.ch.dti)  # pack temperature in °C
        elif self.block.temp_battery is None:
            temp_hor_c = self.block.scenario.temp_air
        else:
            ValueError("Battery temperature must be the name of a PVSource block or numeric")

        temp_hor_k = temp_hor_c + 273.15  # temperature conversion to Kelvin

        # Determine DODs and mean SOCs of (half) cycles within the horizon using the ASTM E 1049-85 norm
        cycles_hor = {"depth": [], "mean": [], "type": []}
        if len(soc_hor) == 2:
            # two timesteps are not enough to detect a cycle -> has to be half cycle by definition -> add manually
            cycles_hor["depth"].append(abs(np.diff(soc_hor)[0]))  # diff of SOCs
            cycles_hor["mean"].append(np.mean(soc_hor))  # mean SOC of cycle
            cycles_hor["type"].append(0.5)  # has to be half cycle
        else:
            for depth, mean, count, _, _ in rainflow.extract_cycles(soc_hor):
                cycles_hor["depth"].append(depth)  # depth of cycle expressed as SOC fraction
                cycles_hor["mean"].append(mean)  # mean SOC of cycle
                cycles_hor["type"].append(count)  # type of cycle 0.5 (half cycle) or 1 (full cycle)
        cycles_hor["depth"] = np.array(cycles_hor["depth"])
        cycles_hor["mean"] = np.array(cycles_hor["mean"])
        cycles_hor["type"] = np.array(cycles_hor["type"])

        # Calculate Number of Full Equivalent Cycles (1 EFC is 2 capacities of charge throughput)
        fec_hor = sum(cycles_hor["type"] * cycles_hor["depth"])
        q_tot_hor = fec_hor * (2 * self.cell.q_nom)

        # Determine actual aging
        self.calc_aging(
            horizon=horizon,
            t_hor=t_hor,
            cycles_hor=cycles_hor,
            temp_hor_k=temp_hor_k,
            ocv_hor=ocv_hor,
            q_tot_hor=q_tot_hor,
            fec_hor=fec_hor,
            crate_hor=crate_hor,
            soc_hor=soc_hor,
        )

        # Update block / block storage size
        self.block.states.loc[horizon.ch.end, "soh"] = 1 - (sum(self.q_loss_cyc) + sum(self.q_loss_cal))
        self.block.states.loc[horizon.ch.end, "q_loss_cal"] = sum(self.q_loss_cal)
        self.block.states.loc[horizon.ch.end, "q_loss_cyc"] = sum(self.q_loss_cyc)
        self.block.states.loc[horizon.ch.end :, "soc_min"] = (1 - self.block.states.loc[horizon.ch.end, "soh"]) / 2
        self.block.states.loc[horizon.ch.end :, "soc_max"] = 1 - (
            (1 - self.block.states.loc[horizon.ch.end, "soh"]) / 2
        )

    @abstractmethod
    def calc_aging(self, horizon, t_hor, cycles_hor, temp_hor_k, **kwargs): ...

    def get_pack_parameters(self):
        self.size = self.block.sizes["storage"].total
        # Calculate number of cells as a float to correctly represent power split with nonreal cells
        self.n_cells = self.size / self.cell.e
        self.m_cells = self.n_cells * self.cell.mass
        self.m_housing = self.m_cells * (1 - self.cell.e_spec_grav_c2p)
        # self.c_th_cells = self.n_cells * self.c_th_cell
        # self.c_th_housing = self.c_th_spec_housing * self.m_housing

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

        i = np.real((-ocv + np.sqrt((ocv**2) + (4 * r_i * p_out))) / (2 * r_i))
        p_loss = r_i * (i**2)

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


class NMCBatteryPackModel(BatteryPackModel):
    # Cell from Schmalstieg et al. - Sanyo UR18650E
    _CELL = "sanyo_ur18650e"

    def calc_aging(self, horizon, t_hor, cycles_hor, temp_hor_k, ocv_hor, q_tot_hor, **kwargs):
        # Schmalstieg aging model is not verified yet against aging data from original paper

        # Set global tuning factor
        k_tuning = 0.43  # Teichert for VW ID.3 cell
        # k_tuning = 1  # deactivation of tuning factor

        #  Calculate calendric stress factor timeseries (from http://dx.doi.org/10.1016/j.jpowsour.2014.02.012)
        alpha_cap = (7.543 * ocv_hor - 23.75) * 1e6 * np.exp(-6976 / temp_hor_k)  # timeseries over all steps
        alpha_res = (5.270 * ocv_hor - 16.32) * 1e5 * np.exp(-5986 / temp_hor_k)  # timeseries over all steps

        # Aggregate calendric stress factors (converting them to scalar) and limit them to zero to avoid
        # a) negative aging and b) problems in calculation of t_eq
        alpha_cap = np.maximum(alpha_cap.mean(), 1e-10)
        alpha_res = np.maximum(alpha_res.mean(), 1e-10)

        # Calculate previous aging state as equivalent time at current conditions
        t_eq_q = (np.sum(self.q_loss_cal) / (k_tuning * alpha_cap)) ** (4 / 3)
        t_eq_r = (np.sum(self.r_inc_cal) / (k_tuning * alpha_res)) ** (4 / 3)

        # Calculate calendric aging in this horizon
        t_hor_days = t_hor / (3600 * 24)  # Schmalstieg model is evaluated in days
        self.q_loss_cal[horizon.index + 1] = k_tuning * alpha_cap * ((t_eq_q + t_hor_days) ** 0.75 - t_eq_q**0.75)
        self.r_inc_cal[horizon.index + 1] = k_tuning * alpha_res * ((t_eq_r + t_hor_days) ** 0.75 - t_eq_r**0.75)

        # Calculate mean OCV of each detected cycle
        ocv_cycles_mean = self.ocv_interp(cycles_hor["mean"]).reshape(
            [
                -1,
            ]
        )
        # Caution: Schmalstieg states quadratic mean (rms) of voltage instead of arithmetic mean!

        # Calculate cyclic stress factor series for each cycle
        beta_cap = 7.348e-3 * (ocv_cycles_mean - 3.667) ** 2 + 7.6e-4 + 4.081e-3 * cycles_hor["depth"]
        beta_res = 2.153e-4 * (ocv_cycles_mean - 3.725) ** 2 - 1.521e-5 + 2.798e-4 * cycles_hor["depth"]
        beta_res = np.maximum(1.5e-5, beta_res)  # limitation as per text following Eq. (21) in paper

        sum_depth = np.sum(cycles_hor["depth"])
        if sum_depth <= 0:
            # no cycling happened
            return

        # actual cycling happened
        # Aggregate cyclic stress factors through DOD-weighted mean (converting them to scalar)
        beta_cap = np.sum(beta_cap * cycles_hor["depth"]) / sum_depth
        beta_res = np.sum(beta_res * cycles_hor["depth"]) / sum_depth

        # Define previous aging state as equivalent FECs at current conditions
        q_eq = (sum(self.q_loss_cyc) / (k_tuning * beta_cap)) ** 2

        # Calculate cyclic aging
        self.q_loss_cyc[horizon.index + 1] = k_tuning * beta_cap * (np.sqrt(q_eq + q_tot_hor) - np.sqrt(q_eq))
        self.r_inc_cyc[horizon.index + 1] = k_tuning * beta_res * q_tot_hor


class LFPBatteryPackModel(BatteryPackModel):
    # Cell from Naumann et al. - Sony US26650
    _CELL = "sony_us26650"

    def calc_aging(self, horizon, t_hor, cycles_hor, temp_hor_k, fec_hor, crate_hor, soc_hor, **kwargs):
        # Set global tuning factor
        k_tuning = 1

        #  Calculate calendric stress factor timeseries (from https://doi.org/10.1016/j.est.2018.01.019)
        k_temp_q_cal = 1.2571e-05 * np.exp((-17126 / 8.3145) * (1 / temp_hor_k - 1 / 298.15))
        k_temp_r_cal = 3.419e-10 * np.exp((-71827 / 8.3145) * (1 / temp_hor_k - 1 / 298.15))
        k_soc_q_cal = 2.85750 * ((soc_hor - 0.5) ** 3) + 0.60225
        k_soc_r_cal = 3.3903 * ((soc_hor - 0.5) ** 2) + 1.56040

        # Aggregate calendric stress factors (converting them to scalar)
        k_temp_q_cal = k_temp_q_cal.mean()
        k_temp_r_cal = k_temp_r_cal.mean()
        k_soc_q_cal = k_soc_q_cal.mean()
        k_soc_r_cal = k_soc_r_cal.mean()

        # Calculate previous aging state as equivalent time at current conditions
        t_eq = np.sum(self.q_loss_cal) / ((k_tuning * k_soc_q_cal * k_temp_q_cal) ** 2)

        # Calculate calendric aging within this horizon
        self.q_loss_cal[horizon.index + 1] = (
            k_tuning * k_temp_q_cal * k_soc_q_cal * (np.sqrt(t_eq + t_hor) - np.sqrt(t_eq))
        )
        self.r_inc_cal[horizon.index + 1] = (
            k_tuning * k_temp_r_cal * k_soc_r_cal * t_hor
        )  # linear - no equivalent time needed

        # Calculate cyclic stress factor series (DOD for each detected cycle, C-rate over time)
        # Methodology from https://doi.org/10.1016/j.jpowsour.2019.227666
        k_dod_q_cyc = 4.0253 * ((cycles_hor["depth"] - 0.6) ** 3) + 1.09230
        k_dod_r_cyc = 6.8477 * ((cycles_hor["depth"] - 0.5) ** 3) + 0.91882
        k_crate_q_cyc = 0.0971 + 0.063 * crate_hor
        k_crate_r_cyc = 0.0023 - 0.0018 * crate_hor

        sum_depth = np.sum(cycles_hor["depth"])
        if sum_depth <= 0:
            # no cycling happened
            return

        # actual cycling happened
        # Aggregate DOD stress factors through DOD-weighted mean (converting them to scalar)
        k_dod_q_cyc = np.sum(k_dod_q_cyc * cycles_hor["depth"]) / sum_depth
        k_dod_r_cyc = np.sum(k_dod_r_cyc * cycles_hor["depth"]) / sum_depth

        # Aggregate C-rate stress factors through arithmetic mean (converting them to a scalar)
        k_crate_q_cyc = k_crate_q_cyc.mean()
        k_crate_r_cyc = k_crate_r_cyc.mean()

        # Define previous aging state as equivalent FECs at current conditions
        fec_eq = 100 * np.sum(self.q_loss_cyc) / ((k_tuning * k_dod_q_cyc * k_crate_q_cyc) ** 2)

        # Calculate cyclic aging within this horizon (0.01 converts percent to fraction)
        self.q_loss_cyc[horizon.index + 1] = (
            0.01 * (k_tuning * k_dod_q_cyc * k_crate_q_cyc) * (np.sqrt(fec_eq + fec_hor) - np.sqrt(fec_eq))
        )
        self.r_inc_cyc[horizon.index + 1] = (
            0.01 * (k_tuning * k_dod_r_cyc * k_crate_r_cyc) * fec_hor
        )  # linear, not fec_eq needed
