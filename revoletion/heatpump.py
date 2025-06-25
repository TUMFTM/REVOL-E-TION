from tespy.components import SimpleHeatExchanger, CycleCloser, Compressor, Valve
from tespy.connections import Connection
from tespy.networks import Network

##define working fluid (propane)
wf = "R290"
nwk = Network(p_unit = "bar", T_unit ="C", iterinfo = False)

#build HP network model
##build components
cp = Compressor("compressor")
ev = SimpleHeatExchanger("evaporator")
cd = SimpleHeatExchanger("condenser")
va = Valve("expansion valve")
cc = CycleCloser("cycle closer")

##build connections
c0 = Connection(va, "out1", cc, "in1", label="0")
c1 = Connection(cc, "out1", ev, "in1", label="1")
c2 = Connection(ev, "out1", cp, "in1", label="2")
c3 = Connection(cp, "out1", cd, "in1", label="3")
c4 = Connection(cd, "out1", va, "in1", label="4")

#connect connections with each other
nwk.add_conns(c0, c1, c2, c3, c4)

#assumption 5K difference for condensation and evaporation --> evaporation at 2°C instead of 7°C and condensation at 40°C instead of 35°C

# connections
c2.set_attr(T=2) #evaporator to compressor
c4.set_attr(T=40) #condenser to valve

# components
cp.set_attr(eta_s=0.8) #efficiency of compressor
cd.set_attr(Q=-9.1e3) #nominal heat delivered by the condeser




