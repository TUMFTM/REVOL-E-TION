import pandas as pd
import os


class EPF:
    def __init__(self, model_name):
        self.model = model_name
        self.costs = pd.DataFrame()

        path = './input/grid/'
        dataset = 'ALL'
        file_path = os.path.join(path, dataset + '.csv')
        # Read data; already parse dates and convert prices from €/MWh to €/Wh
        self.data = pd.read_csv(file_path, index_col=0, parse_dates=True, float_precision='round_trip') * 1e-6

        # Rename columns
        rename_dict = {'Real_15minutes': 'Real',
                       'LEAR_15minutes': 'statistical',
                       'DNN_15minutes': 'dnn',
                       'Actual_15minutes': 'actual'}
        self.data = self.data.rename(columns=rename_dict)

        # ToDo: seems to be unnecessary as data is accessed by column name in the following
        if self.model == 'statistical':
            self.data = self.data[['Real', 'statistical']]
        elif self.model == 'dnn':
            self.data = self.data[['Real', 'dnn']]
        elif self.model == 'actual':
            self.data = self.data[['Real', 'actual']]

    def update_costs(self, ph_dti, costs):
        # ToDo: update cost series for the ph_dti time index with the real day ahead prices
        #  for day X+1 and the predicted prices for day X+2

        start_date = ph_dti[0]

        day1_range = pd.date_range(start_date, start_date + pd.Timedelta(days=1)- pd.Timedelta(minutes=15), freq='15T')
        day2_range = pd.date_range(start_date + pd.Timedelta(days=1), start_date + pd.Timedelta(days=2) - pd.Timedelta(minutes=15), freq='15T')

        day1_range = day1_range[day1_range.isin(costs.index)]
        day2_range = day2_range[day2_range.isin(costs.index)]

        costs.loc[day1_range] = self.data.loc[day1_range, 'Real']
        costs.loc[day2_range] = self.data.loc[day2_range, self.model]

        # you can specify the name of the models here. Just make sure to use the same name in the sgen file
        #letzter tag von day 2 range muss noch in kosten vektor sein

        return costs

    def get_flow(self, flow_in, flow_out, scenario):
        # flow_in = flow from grid to local grid
        # flow_out = flow from local grid to grid
        # ToDo: Use these flows to calculate the costs or revenues related to the energy flow from and to the grid
        #  print or save the results
        overall_result = pd.DataFrame(index=scenario.sim_dti)

        # Alternative approach instead of whole if/else structure
        # Costs * Powerflow * Timestep (conversion from power to energy) -> a single value for the whole simulation

        # result = (self.data.loc[scenario.sim_dti, self.model] * flow_out * scenario.timestep_hours).sum()
        # print(f'Costs for scheduling based on {self.model} costs: {result} €')

        # ToDo: manually create a new file in results called results_epf.txt -> csv is not necessary for single values

        # with open('results/results_epf.txt', 'a') as file:
        #     file.write(f'Costs for scheduling based on {self.model} costs: {result} €\n')

        if self.model == 'statistical':
            self.data = self.data[['statistical']]
            result_statistical = self.data.loc[scenario.sim_dti] * (flow_in + flow_out)  # * (1/1000000) not needed anymore
            result_statistical = result_statistical.sum()
            print(result_statistical)
            overall_result['result_statistical'] = result_statistical
            path_result = './results/'
            overall_result.to_csv(os.path.join(path_result, 'statistical_result.csv'))
            pass
        elif self.model == 'dnn':
            self.data = self.data[['dnn']]
            result_dnn = self.data.loc[scenario.sim_dti] * (flow_in + flow_out)  # * (1/1000000) not needed anymore
            result_dnn = result_dnn.sum()
            print(result_dnn)
            overall_result['result_dnn'] = result_dnn
            path_result = './results/'
            overall_result.to_csv(os.path.join(path_result, 'dnn_result.csv'))
            pass
        elif self.model == 'actual':
            self.data = self.data[['actual']]
            result_actual = self.data.loc[scenario.sim_dti] * (flow_in + flow_out)  # * (1/1000000) not needed anymore
            result_actual = result_actual.sum()
            print(result_actual)
            overall_result['result_actual'] = result_actual
            path_result = './results/'
            overall_result.to_csv(os.path.join(path_result, 'actual_result.csv'))
            pass

        #path_result = './results/'
        #if not os.path.exists(path_result):
         #   os.makedirs(path_result)
        #overall_result.to_csv(os.path.join(path_result,'overall_result.csv'))

        #self.data gehen und diese Daten verwenden
        #sim_dti Parameter in scenario - beinhaltet gesamten Zeitraum der simulation

        # ToDo: remove return -> won't be used in the main file as function is conceptualized as void
        return overall_result