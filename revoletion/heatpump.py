from tespy.components import SimpleHeatExchanger, CycleCloser, Compressor, Valve
from tespy.connections import Connection
from tespy.networks import Network
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class Heatpump_COPanalyzer:
    def __init__(self,
                 wf: str = "R290",
                 nominal_cop: float = 4.9,
                 nominal_power: float = 9100,
                 T_W35: float = 35,
                 T_A7: float = 7,
                 T_spread: float = 5,
                 ):

        fluid_map = {
            'r290': 'R290',
            'r600a': 'R600a',
            'r1234yf': 'R1234yf',
            'r744': 'R744',
        }

        self.wf = fluid_map.get(wf.lower(), None)
        if not self.wf:
            raise ValueError(f"Unknown working fluid: '{wf}'. Valid options are: {', '.join(fluid_map.keys())}")

        self.nominal_cop = nominal_cop
        self.nominal_power = nominal_power
        self.T_W35 = T_W35
        self.T_A7 = T_A7
        self.T_spread = T_spread
        self.build_heatpump()
        self.results = None

    def build_heatpump(self):
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
        self.c2.set_attr(T=self.T_A7-self.T_spread, fluid={self.wf: 1}, x=1.0)  # evaporator to compressor
        self.c4.set_attr(T=self.T_W35+self.T_spread, x=0.0)  # condenser to valve

        # components
        self.cp.set_attr(eta_s=0.8)  # efficiency of compressor
        self.cd.set_attr(Q=(-1) * self.nominal_power, pr = 0.98)  # nominal heat delivered by the condenser and loss assumption
        self.ev.set_attr(pr=0.99)  # loss assumption

        # solve network
        self.nwk.solve("design")

    def cop_optimization(self, max_iter = 10):
        eta_s_max = 0.8
        eta_s_min = 0.4

        for _ in range(max_iter):
            eta_s = (eta_s_max+eta_s_min) / 2
            self.cp.set_attr(eta_s=eta_s)
            self.nwk.solve("design")
            COP = abs(self.cd.Q.val)/self.cp.P.val

            if round(COP - self.nominal_cop, 3) > 0:
                eta_s_max = eta_s
            elif round(COP - self.nominal_cop, 3) < 0:
                eta_s_min = eta_s
            else:
                break

        self.efficiency = round(self.cp.eta_s.val,3)

    def calculate_cop(self, temperature_range=np.arange(-10, 21)):
        self.temperature_range = temperature_range
        results = pd.DataFrame(index=temperature_range, columns=["COP", "COP Carnot"])

        for T in temperature_range:
            self.c2.set_attr(T=T - self.T_spread)
            self.nwk.solve("design")
            results.loc[T, "COP"] = abs(self.cd.Q.val) / self.cp.P.val
            results.loc[T, "COP_carnot"] = self.c4.T.val_SI / (self.c4.T.val - self.c2.T.val)

        results["efficiency"] = results["COP"] / results["COP_carnot"]
        self.results = results
        return results

    def analyze_cop(self, temperature_range=np.arange(-10, 21)):
        self.temperature_range = temperature_range
        results = pd.DataFrame(index=temperature_range, columns=["COP", "COP_carnot"])

        for T in temperature_range:
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
        self.build_heatpump()
        self.cop_optimization()
        self.analyze_cop()
        return self.get_cop_array()

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



