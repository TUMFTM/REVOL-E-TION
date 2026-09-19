import pandas as pd

from revoletion.economics import model as ecoclasses
from revoletion.economics import params

eco = params.EcoParams.from_parameters(
    prj_duration_yrs=20,
    discount_rate=0.05,
    compensate_sim_prj=True,
    dti_sim=pd.date_range(start="2024-01-01", end="2025-01-01", freq="15min", tz="Europe/Berlin", inclusive="left"),
    dti_eval=pd.date_range(start="2024-02-01", end="2024-03-01", freq="15min", tz="Europe/Berlin", inclusive="left"),
)

agg = ecoclasses.Aggregator.create(name="agg", eco=eco)

evalulator = ecoclasses.Evaluator.create(
    name="block",
    eco=eco,
    data_dir=None,
    capex=params.CapexParams(
        spec=1.0,
        fix=2000,
        ls=10,
    ),
    mntex=params.MntexParams(
        spec=1.0,
        fix=2000,
    ),
    opex=params.OpexParams(
        spec_energy=0.5,
        fix=10,
    )
)

agg.add_node(evalulator)

evalulator.evaluate(size_preexisting=10, size_expansion=0, power=pd.Series(index=eco.dti_eval, data=0.001))
agg.aggregate()
pass