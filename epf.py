import pandas as pd
import os


class EPF:
    def __init__(self, model_name, file_path, data_name):
        self.model = model_name
        self.file_path = file_path
        self.data_name = data_name
        self.costs = pd.DataFrame()
        ##daten einlesen (gesamte datei)
        #self cost rea
        #self cost predicted

        self.data = pd.read_csv(os.path.join(file_path, data_name + '.csv'), index_col=0, parse_dates=True)
        self.data.columns = ['Real','Predict']

    def update_costs(self, ph_dti, costs):
        # ToDo: update cost series for the ph_dti time index with the real day ahead prices
        #  for day X+1 and the predicted prices for day X+2

        self.costs = costs

        for date in ph_dti:
            day1 = self.data.loc[date:date + pd.Timedelta(days=1)]
            day2 = self.data.loc[date + pd.Timedelta(days=1):date + pd.Timedelta(days=2)]

            costs.loc[day1.index, 'Cost1'] = day1['Real']
            costs.loc[day2.index, 'Cost2'] = day2['Predict']

        # you can specify the name of the models here. Just make sure to use the same name in the sgen file
        if self.model == 'DNN':
            pass
        elif self.model == 'LEAR':
            pass
        elif self.model == 'real':
            pass

        return costs

    def get_flow(self, flow_in, flow_out):
        # flow_in = flow from grid to local grid
        # flow_out = flow from local grid to grid
        # ToDo: Use these flows to calculate the costs or revenues related to the energy flow from and to the grid
        #  print or save the results
        result = self.costs * (flow_in + flow_out)
        print(result)

        return result