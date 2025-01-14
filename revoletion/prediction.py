


class MobilityPrediction:
    def __init__(self, parent):
        self.parent = parent
        self.scenario = self.parent.scenario

    def predict(self, horizon):
        # ToDo: insert code for prediction here
        # Dummy prediction code -> replace
        data_predicted = self.parent.data.loc[horizon.starttime:horizon.ph_endtime, :]

        # update data of commodities
        for commodity in self.parent.commodities.values():
            commodity.data_ph = data_predicted.loc[:, (commodity.name, slice(None))].droplevel(0, axis=1)
