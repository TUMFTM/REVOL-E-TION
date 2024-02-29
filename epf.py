import pandas as pd
import os


class EPF:
    def __init__(self, model_name, scenario):
        self.model = model_name
        self.costs = pd.DataFrame()

        dataset = f'Forecast_{scenario.sim_dti[0].year}.csv'
        file_path = os.path.join('input', 'grid', dataset)
        # Read data; already parse dates and convert prices from €/MWh to €/Wh
        self.data = pd.read_csv(file_path, index_col=0, parse_dates=True) * 1e-6

        # Rename columns
        rename_dict = {'Real_15minutes': 'real',
                       'LEAR_15minutes': 'statistical',
                       'DNN_15minutes': 'dnn'}
        self.data = self.data.rename(columns=rename_dict)

        # ToDo: seems to be unnecessary as data is accessed by column name in the following

    def update_costs(self, ph_dti, costs):
        # ToDo: update cost series for the ph_dti time index with the real day ahead prices
        #  for day X+1 and the predicted prices for day X+2

        start_date = ph_dti[0]

        day1_range = pd.date_range(start_date, start_date + pd.Timedelta(days=1)- pd.Timedelta(minutes=15), freq='15T')
        day2_range = pd.date_range(start_date + pd.Timedelta(days=1), ph_dti[-1], freq='15T')

        day1_range = day1_range[day1_range.isin(costs.index)]
        day2_range = day2_range[day2_range.isin(costs.index)]

        costs.loc[day1_range] = self.data.loc[day1_range, 'real']
        costs.loc[day2_range] = self.data.loc[day2_range, self.model]

        # you can specify the name of the models here. Just make sure to use the same name in the sgen file
        #letzter tag von day 2 range muss noch in kosten vektor sein

        return costs

    def get_flow(self, flow_in, flow_out, scenario):
        # flow_in = flow from grid to local grid
        # flow_out = flow from local grid to grid
        # ToDo: Use these flows to calculate the costs or revenues related to the energy flow from and to the grid
        #  print or save the results
        #overall_result = pd.DataFrame(index=scenario.sim_dti)

        # Alternative approach instead of whole if/else structure
        # Costs * Powerflow * Timestep (conversion from power to energy) -> a single value for the whole simulation
        tax_balance = (flow_out* scenario.timestep_hours).sum()*14.634*1e-5
        energy_balance = (self.data.loc[scenario.sim_dti, 'real'] * flow_out * scenario.timestep_hours).sum()
        result = (tax_balance+energy_balance)*1.19

        print(f'Taxes excl. VAT for scheduling based on {self.model} costs: {tax_balance} €')
        print(f'NettoCosts for scheduling based on {self.model} costs: {energy_balance} €')
        print(f'Costs for scheduling based on {self.model} costs: {result} €')

        # ToDo: manually create a new file in results called results_epf.txt -> csv is not necessary for single values

        with open('results/results_epf.txt', 'a') as file:
            file.write(f'Costs for scheduling based on {self.model} costs: {result} €\n')

        # ToDo: remove return -> won't be used in the main file as function is conceptualized as void