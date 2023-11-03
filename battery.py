# Import libraries
import numpy as np
import scipy.interpolate as spip
import pickle
import os


class BatteryCell:

    def __init__(self, chemistry):

        self.chemistry = chemistry.lower()
        self.soc_max = 1  # Upper SOC limit
        self.soc_min = 0  # Lower SOC limit
        self.c_th = 76.27  # Cell heat capacity as per Forgez et al. # todo find unit
        self.R_th_in = 3.3  # Thermal resistance between cell and Housing # todo find unit

        if self.chemistry == 'nmc':
            # Cell from Schmalstieg et al.
            self.q_nom = 2.05  # Nominal capacity in Ah
            self.u_nom = 3.6  # Nominal voltage in V
            self.u_min = 3.0  # Minimum voltage in V
            self.u_max = 4.2  # Maximum voltage in V
            self.i_max_cont = 2.05  # Maximum charging current in A
            self.i_min_cont = -6.15  # Maximum discharging current in A
            self.mass = 0.0445  # Cell mass in kg
            # self.e_spec_grav = bet.gravimetric_energy_density[Chemistry]
            # self.c2p_grav = bet.c2p_grav[Chemistry]
            self.data_path = os.path.join(os.getcwd(), 'input', 'battery', 'sony_us26650.pkl')

        elif self.chemistry == 'lfp':
            # Cell from Naumann et al.
            self.q_nom = 3.0  # Nominal capacity in Ah
            self.u_nom = 3.2  # Nominal voltage in V
            self.u_min = 2.0  # Minimum voltage in V
            self.u_max = 3.6  # Maximum voltage in V
            self.i_max_cont = 3  # Maximum charging current in A
            self.i_min_cont = -20  # Maximum discharging current in A
            self.mass = 0.0845  # Cell mass in kg
            #self.e_spec_grav = bet.gravimetric_energy_density[Chemistry]
            #self.c2p_grav = bet.c2p_grav[Chemistry]
            self.data_path = os.path.join(os.getcwd(), 'input', 'battery', 'sony_us26650.pkl')

        with open(self.data_path, 'rb') as file:
            self.ocv, self.r_i_ch, self.r_i_dch = pickle.load(file)

        #self.ocv_func = spip.interp1d(U_ocv_data[:, 0], U_ocv_data[:, 1], kind="linear")
        #self.r_i_cch_func = spip.interp2d(x_value, y_value, R_i_ch_table, kind="linear")
        #self.r_i_dch_func = spip.interp2d(x_value, y_value, R_i_dch_table, kind="linear")

    def rint_model(self, p_out):
        
        ocv = self.ocv_func(self.soc)
        if p_out > 0:
            r_i = self.r_i_ch_func(self.t_cell[-1], self.soc[-1])
        else:
            r_i_dch = self.r_i_dch_func(self.t_cell[-1], self.soc[-1])

        i = np.real((-ocv + np.sqrt((ocv ** 2) + (4 * r_i * p_out))) / (2 * r_i))
        p_loss = (i ** 2) * r_i

        return i, p_loss






# Setting Battery Model type (2) Schmalstieg
def bat_ss_setting(cell):


    # Internal Resistance
    # Internal Resistance
    Ri_data = np.genfromtxt("CellData/NMC_Sanyo_Rint.csv", delimiter=',')
    R_i_ch_table = Ri_data[1:, 2:6]
    R_i_dch_table = Ri_data[1:, 6:10]
    x_value = [10, 25, 40, 60]
    y_value = np.arange(0, 1.01, 0.01)


    cell.ocv_func = interp1d(points, u_ocv, kind="linear")

    return cell


