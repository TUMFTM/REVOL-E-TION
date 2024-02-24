import pandas as pd
import os


class EPF:
    def __init__(self, model_name):
        self.model = model_name
        self.costs = pd.DataFrame()

        path = './input/grid/'
        dataset = 'ALL'
        file_path = os.path.join(path, dataset + '.csv')
        self.data = pd.read_csv(file_path, index_col=0, parse_dates=True, float_precision='round_trip')
        self.data.columns = ['Real', 'statistical', 'dnn', 'actual']

        if self.model == 'statistical':
            self.data = self.data[['Real', 'statistical']]
            pass
        elif self.model == 'dnn':
            self.data = self.data[['Real', 'dnn']]
            pass
        elif self.model == 'actual':
            self.data = self.data[['Real', 'actual']]
            pass

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

        if self.model == 'statistical':
            self.data = self.data[['statistical']]
            result_statistical = self.data.loc[scenario.sim_dti] * (1/1000000)*(flow_in + flow_out)
            result_statistical = result_statistical.sum()
            print(result_statistical)
            overall_result['result_statistical'] = result_statistical
            pass
        elif self.model == 'dnn':
            self.data = self.data[['dnn']]
            result_dnn = self.data.loc[scenario.sim_dti] *(1/1000000)* (flow_in + flow_out)
            result_dnn = result_dnn.sum()
            print(result_dnn)
            overall_result['result_dnn'] = result_dnn
            pass
        elif self.model == 'actual':
            self.data = self.data[['actual']]
            result_actual = self.data.loc[scenario.sim_dti]*(1/1000000)* (flow_in + flow_out)
            result_actual = result_actual.sum()
            print(result_actual)
            overall_result['result_actual'] = result_actual
            pass

        path_result = './results/'
        if not os.path.exists(path_result):
            os.makedirs(path_result)
        overall_result.to_csv(os.path.join(path_result,'overall_result.csv'))

        #self.data gehen und diese Daten verwenden
        #sim_dti Parameter in scenario - beinhaltet gesamten Zeitraum der simulation

        return overall_result