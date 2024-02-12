import pandas as pd


class EPF:
    def __init__(self, model_name):
        self.model = model_name

    def update_costs(self, ph_dti, costs):
        # ToDo: update cost series for the ph_dti time index with the real day ahead prices
        #  for day X+1 and the predicted prices for day X+2
        # you can specify the name of the models here. Just make sure to use the same name in the sgen file
        if self.model == 'dl':
            pass
        elif self.model == 'statistical':
            pass
        elif self.model == 'real':
            pass

        return costs

    def get_flow(self, flow_in, flow_out):
        # flow_in = flow from grid to local grid
        # flow_out = flow from local grid to grid
        # ToDo: Use these flows to calculate the costs or revenues related to the energy flow from and to the grid
        #  print or save the results
        return