


class MobilityPrediction:
    def __init__(self, parent):
        self.parent = parent
        self.scenario = self.parent.scenario

    def predict(self, horizon):
        # ToDo: insert code for prediction here

        # Prediction code

        return self.parent.data.loc[horizon.starttime:horizon.ph_endtime]