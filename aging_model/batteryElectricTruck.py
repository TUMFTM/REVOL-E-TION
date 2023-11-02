"""
Created on Fri Aug 27 16:32:19 2021

@author: ga78tas
"""


class BatteryElectricTruck():
    
    def __init__(self):

        # Battery Thermal System
        self.Pcooler = 10e3  # Cooling power in W [Schimpe et al.]
        self.Pheater = 11.2e3  # Heating power in W [Schimpe et al.]
        self.COPcool = -3 #Coefficient of performance of cooling system in pu [Schimpe et al.]
        self.COPheat = 4  # Coefficient of performance of heating system in pu [Schimpe et al., Danish Energy Agency 2012]
        self.Ebat_Neubauer = 22.1e3  # Battery size used by Neubauer et al. in Wh [Neubauer et al.]
        self.Rth_Neubauer = 4.343  # Thermal resistance between housing and ambient used by Neubauer et al. in W/K [Neubauer et al.]
        self.Cth_Neubauer = 182e3  # Thermal mass of battery in J/K [Neubauer]
        self.T_Cool_on = 33 # Activation Cooling Threshold [Neubauer et al.]
        self.T_Cool_off = 31.5 # Deactivation Cooling Threshold [ID3 Paper]
        self.T_Heat = 10  # Heating Threshold [Neubauer et al.]
        self.k_bh = 0.899 # K/W #OlafDiss
        self.c_aluminium = 896 # J/(kgK)#OlafDiss
        self.k_out = 10.9 #W/K OlafDiss
        self.gravimetric_energy_density = [273, 176]  # in Wh/kg
        self.volumetric_energy_density = [685, 376]  # in Wh/l
        self.c2p_grav = [0.59, 0.71]
        self.c2p_vol = [0.39, 0.55]

        