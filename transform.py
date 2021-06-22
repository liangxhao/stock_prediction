import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Transform(object):
    def __init__(self):
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def fit_transform(self, data):
        data = np.square(data)
        return self.scaler.fit_transform(data)

    def transform(self, data):
        data = np.square(data)
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        data = self.scaler.inverse_transform(data)
        data = np.sqrt(data)
        return data