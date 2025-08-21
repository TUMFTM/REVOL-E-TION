from tespy.components import SimpleHeatExchanger, CycleCloser, Compressor, Valve
from tespy.connections import Connection
from tespy.networks import Network
import pandas as pd
import numpy as np
from demandlib import vdi
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

T_W35 = 35
T_A7= 7
T_SPREAD = 5


class Heatpump_COPanalyzer:
    def __init__(self,
                 working_fluid: str = "R290",
                 nominal_cop: float = 4.9,
                 nominal_power: float = 9100,
                 temperature_range=np.arange(-10, 21)
                 ):

        fluid_map = {
            'r290': 'R290',
            'r600a': 'R600a',
            'r1234yf': 'R1234yf',
            'r744': 'R744',
        }

        self.working_fluid = fluid_map.get(working_fluid.lower(), None)
        if not self.working_fluid:
            raise ValueError(f"Unknown working fluid: '{working_fluid}'. Valid options are: {', '.join(fluid_map.keys())}")

        self.nominal_cop = nominal_cop
        self.nominal_power = nominal_power
        self.T_W35 = T_W35
        self.T_A7 = T_A7
        self.T_spread = T_SPREAD
        self.temperature_range = temperature_range

        self.nwk = Network(p_unit="bar", T_unit="C", iterinfo=False)

        # build HP network model
        ##build components
        self.cp = Compressor("compressor")
        self.ev = SimpleHeatExchanger("evaporator")
        self.cd = SimpleHeatExchanger("condenser")
        self.va = Valve("expansion valve")
        self.cc = CycleCloser("cycle closer")

        ##build connections
        self.c0 = Connection(self.va, "out1", self.cc, "in1", label="0")
        self.c1 = Connection(self.cc, "out1", self.ev, "in1", label="1")
        self.c2 = Connection(self.ev, "out1", self.cp, "in1", label="2")
        self.c3 = Connection(self.cp, "out1", self.cd, "in1", label="3")
        self.c4 = Connection(self.cd, "out1", self.va, "in1", label="4")

        # connect connections with each other
        self.nwk.add_conns(self.c0, self.c1, self.c2, self.c3, self.c4)

        # connections
        self.c2.set_attr(T=self.T_A7 - self.T_spread, fluid={self.working_fluid: 1}, x=1.0)  # evaporator to compressor
        self.c4.set_attr(T=self.T_W35 + self.T_spread, x=0.0)  # condenser to valve

        # components
        self.cp.set_attr(eta_s=0.8)  # efficiency of compressor
        self.cd.set_attr(Q=(-1) * self.nominal_power,
                         pr=0.98)  # nominal heat delivered by the condenser and loss assumption
        self.ev.set_attr(pr=0.99)  # loss assumption

        # solve network
        self.nwk.solve("design")

        self.results = None

    def cop_optimization(self, max_iter = 10):
        eta_s_max = 0.8
        eta_s_min = 0.4

        for _ in range(max_iter):
            eta_s = (eta_s_max+eta_s_min) / 2
            self.cp.set_attr(eta_s=eta_s)
            self.nwk.solve("design")
            COP = abs(self.cd.Q.val)/self.cp.P.val

            if COP - self.nominal_cop > 0:
                eta_s_max = eta_s
            elif COP - self.nominal_cop < 0:
                eta_s_min = eta_s
            else:
                break

        self.efficiency = round(self.cp.eta_s.val, 3)

    def analyze_cop(self):
        results = pd.DataFrame(index=self.temperature_range, columns=["COP", "COP_carnot"])

        for T in self.temperature_range:
            self.c2.set_attr(T=T - self.T_spread)
            self.nwk.solve("design")
            results.loc[T, "COP"] = abs(self.cd.Q.val) / self.cp.P.val
            results.loc[T, "COP_carnot"] = self.c4.T.val_SI / (self.c4.T.val - self.c2.T.val)

        results["efficiency"] = results["COP"] / results["COP_carnot"]
        self.results = results
        return results

    def get_cop_array(self):
        if self.results is None:
            raise ValueError("Run analyze_cop first.")

        coarse_temps = self.results.index
        coarse_cops = self.results["COP"]

        temp_start, temp_stop = coarse_temps.min(), coarse_temps.max()
        fine_temps = np.round(np.arange(temp_start, temp_stop + 0.01, 0.01), 2)

        interp_func = interp1d(coarse_temps, coarse_cops, kind="linear", fill_value="extrapolate")
        fine_cops = np.round(interp_func(fine_temps), 2)

        self.results = pd.DataFrame(data={"COP": fine_cops}, index=fine_temps)

        return self.results["COP"]

    def run_full_analysis(self):
        self.cop_optimization()
        self.analyze_cop()
        return self.get_cop_array()

    def get_heating_energy_apriori(self, scenario, size_household, type_house, size_house,
                                   demand_spec, temperature_tolerance,
                                   flows_apriori, states, cop_array):
        type_house_map = {
            "efh": "EFH",
            "mfh": "MFH"
        }

        type_house_upper = type_house_map.get(type_house.lower(), type_house)

        temp_air = scenario.temp_air

        houses = [
            {
                "name": f"{type_house_upper}_1",
                "house_type": type_house_upper,
                "N_Pers": size_household,
                "N_WE": 1,
                "Q_Heiz_a": size_house * demand_spec,
                "Q_TWW_a": size_household * 500000 if type_house_upper == "EFH" else
                   size_household * 1000000 if type_house_upper == "MFH" else
                   0,
                "W_a": 0,
                "summer_temperature_limit": 15,
                "winter_temperature_limit": 5,
            }
        ]

        try_region = vdi.find_try_region(scenario.longitude, scenario.latitude)
        demand_list = []

        for year in scenario.temp_air.index.year.unique():
            region = vdi.Region(
                year=year,
                climate=vdi.Climate().from_try_data(try_region),
                houses=houses,
                resample_rule=scenario.timestep_td
            )

            demand_year = region.get_load_curve_houses().iloc[:, :2]
            demand_year.columns = ['demand_heat', 'demand_dhw']
            demand_list.append(demand_year)

        demand_accumulated = pd.concat(demand_list, axis=1)
        demand_accumulated.index = demand_accumulated.index + pd.DateOffset(hours=-1)
        demand_accumulated.index = demand_accumulated.index.tz_localize("UTC")
        demand_accumulated.index = demand_accumulated.index.tz_convert('Europe/Berlin')
        demand_accumulated = demand_accumulated.loc[scenario.temp_air.index]

        # Direkte Zuweisung in flows_apriori
        flows_apriori['demand_heat'] = demand_accumulated['demand_heat']
        flows_apriori['demand_dhw'] = demand_accumulated['demand_dhw']

        # Nachtabschaltung
        mask_night =  (scenario.temp_air.index.hour >= 22) | (scenario.temp_air.index.hour <= 6)

        # Thermische Trägheit

        flows_apriori['delta'] = flows_apriori['demand_heat'] * (
                1 - np.where(
                (20 - temp_air['temp_air']) != 0,
                (20 - temperature_tolerance - temp_air['temp_air'])/ (20 - temp_air['temp_air']),
                0)
        )

        flows_apriori['min'] = flows_apriori['demand_heat']-flows_apriori['delta']
        flows_apriori['max'] = flows_apriori['demand_heat'] + flows_apriori['delta']
        flows_apriori['dif'] = flows_apriori['max']-flows_apriori['min']

        # COP-Werte
        cop_series = temp_air['temp_air'].round(2).map(cop_array)
        cop_series.loc[temp_air['temp_air'] > 20] = cop_array.max()
        states['COP'] = cop_series

    def plot_results(self, T_for_eta=7, save_path=None):
        if self.results is None:
            raise ValueError("No results available. Run analyze_cop_vs_temperature first.")

        eta_const = self.results.loc[T_for_eta, "efficiency"]
        fig, ax = plt.subplots(2, sharex=True)

        ax[0].plot(self.temperature_range, self.results["COP_carnot"], label="COPₙ")
        ax[0].plot(self.temperature_range, self.results["COP"], label="COP")
        label = f"$\\mathrm{{COP}}$: $\\eta(T={T_for_eta}°C)={round(eta_const, 3)}$"
        ax[0].plot(self.temperature_range, self.results["COP_carnot"] * eta_const, label=label)
        ax[0].set_ylabel("COP")
        ax[0].legend()

        ax[1].plot(self.temperature_range, self.results["efficiency"], color="tab:orange")
        ax[1].plot(self.temperature_range, [eta_const] * len(self.temperature_range), color="tab:green")
        ax[1].set_ylabel("Efficiency factor")
        ax[1].set_xlabel("Ambient temperature in °C")

        for a in ax:
            a.grid()
            a.set_axisbelow(True)

        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()



