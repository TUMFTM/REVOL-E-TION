# ToDo MGEV-Tool

### Prio (Urgent changes)

### Feature Development

- Enabling nearly realistic modelling of EV charging path <mark>Philipp Rosner
  - Use oemof OffsetConverter as individual commodity charge/discharge converters
  - predefine two normalized efficiency curves for AC and DC charge path
  - Scale efficiency behavior to actual charge power
  - Convert efficiency behavior to linear in/output power relation within a function
  - Conventional Efficiencies equal c0=0, c1=eff
- Enable AC/DC switching of CommoditySystem connection to system core <mark>Philipp Rosner
- Integrate simplified battery degradation analysis post-operation <mark>Philipp Rosner
  - Take methodology from Max Zähringer / Jakob Schneider for LFP cells (Naumann et al)
- Enabling "real" V2G not only into Minigrid but into external grid <mark>Brian Dietermann
  - Add structure in scenario definition 
  - Add additional sink next to grid connection
  - Unify wording and meaning of v2g -> Suggestion: uc < cc < v2v < v2mg < v2g
- Enabling external charging <mark>Brian Dietermann
  - Add structure in scenario definition
  - Add additional source for each vehicle
  - Concept for designing vehicle input data -> additional column "external_charging" needed
  - Track external charging energies and cost separately through model evaluation for as long as possible to enable evaluation of different business models
- Add further technoeconomic evaluation metrics <mark> Philipp Rosner
  - Internal rate of Return
  - RE curtailment
  - stationary ESS energy throughput

### Adaptions
- Convert all time (indices) used to UTC instead of local time
  - Make corresponding entry into readme
- Resampling is not working for data which stops before scenario.sim_endtime (e.g. if last data entry is at 23:00 resampling to 15T doesn't work as last entry then should be at 23:45):  
  - Use standard resampling -> new DateTimeIndex[-1] <= old DateTimeIndex[-1]  
  - When upsampling new values have to be added at the end (e.g. last: 03:00 -> 15T -> last: 03:45)  
  - In case of new DateTimeIndex[-1] != old DateTimeIndex[-1] (equal to old timespan % new freq !=0) ffill() doesn't work  
  - Manually use last value of old data (no more NaNs due to ffill and bfill)  
  - -> df = df.resample(new_freq, axis=0).mean().ffill().bfill(). reindex(pd.date_range(start=df.index.min(), end=end_t, freq=new_freq, inclusive="left")).fillna(df.iloc[-1, :])  

### Bugfixing
- Hourly time steps produce pandas errors
- Large scenario files (>ca. 100 scenarios) lead to failures in joining up scenarios

### Tests
- Test whether units of specific opex values are per timestep or per hour <mark>Philipp Rosner
- Test behavior of different min SOCs with DES & optimization techniques

### Done
- Enabling time-variable OpEx (e.g. for grid electricity prices) <mark>Brian Dietermann
  - change cost calculation in blocks/InvestBlock/calc_eco_results from (energy times constant price) to (flow timeseries inner product with cost timeseries)
  - requires conversion of scalar cost values to vectors
  - requires decision on how to handle cost vectors in input json file
  - requires decision for which classes besides ControllableSource the feature has to be implemented

