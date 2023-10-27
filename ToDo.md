# ToDo MGEV-Tool

### Prio (Urgent changes)

### Feature Development
- Enabling time-variable OpEx (e.g. for grid electricity prices) <mark>Brian Dietermann
  - change cost calculation in blocks/InvestBlock/calc_eco_results from (energy times constant price) to (flow timeseries inner product with cost timeseries)
  - requires conversion of scalar cost values to vectors
  - requires decision on how to handle cost vectors in input json file
- Enabling nearly realistic modelling of EV charging path <mark>Philipp Rosner
  - Use oemof OffsetConverter as individual commodity charge/discharge converters
  - predefine two normalized efficiency curves for AC and DC charge path
  - Scale efficiency behavior to actual charge power
  - Convert efficiency behavior to linear in/output power relation within a function
  - Conventional Efficiencies tauchen nur in den Komponentendefinitionen auf, nicht in nachgelagerten Berechnungen
- Enable AC/DC switching of CommoditySystem connection to system core <mark>Philipp Rosner
- Integrate simplified battery degradation analysis post-operation
  - Take methodology from Max Zähringer / Jakob Schneider for LFP cells (Naumann et al)

### Adaptions
- Convert all time (indices) used to UTC instead of local time
  - Make corresponding entry into readme

### Bugfixing
- Hourly time steps produce pandas errors
- Large scenario files (>ca. 100 scenarios) lead to failures in joining up scenarios

### Tests
- Test whether units of specific opex values are per timestep or per hour <mark>Philipp Rosner

