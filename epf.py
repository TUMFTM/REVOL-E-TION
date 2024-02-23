import pandas as pd
import os


class EPF:
    def __init__(self, model_name):
        self.model = model_name
        self.costs = pd.DataFrame()

        path = './input/grid/'
        dataset = 'ALL'
        file_path = os.path.join(path, dataset + '.csv')
        self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        self.data.columns = ['Real', 'statistical', 'dnn']

        if self.model == 'statistical':
            self.data = self.data[['Real', 'statistical']]
            pass
        elif self.model == 'dnn':
            self.data = self.data[['Real', 'dnn']]
            pass
        #elif self.model == 'real':
            #pass

    def update_costs(self, ph_dti, costs):
        # ToDo: update cost series for the ph_dti time index with the real day ahead prices
        #  for day X+1 and the predicted prices for day X+2

        self.costs = costs
        start_date = ph_dti[0]

        day1_range = pd.date_range(start_date, start_date + pd.Timedelta(days=1)- pd.Timedelta(minutes=15), freq='15T')
        day2_range = pd.date_range(start_date + pd.Timedelta(days=1), start_date + pd.Timedelta(days=2) - pd.Timedelta(minutes=15), freq='15T')
        costs.loc[day1_range] = self.data.loc[day1_range, 'Real']
        costs.loc[day2_range] = self.data.loc[day2_range, self.model]

        # you can specify the name of the models here. Just make sure to use the same name in the sgen file

        return costs

    def get_flow(self, flow_in, flow_out):
        # flow_in = flow from grid to local grid
        # flow_out = flow from local grid to grid
        # ToDo: Use these flows to calculate the costs or revenues related to the energy flow from and to the grid
        #  print or save the results
        result = self.costs * (flow_in + flow_out)
        result = result.sum()
        print(result)

        return result