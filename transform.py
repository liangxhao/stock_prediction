import numpy as np
from sklearn.preprocessing import StandardScaler


class Transform(object):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, data):
        x = np.sqrt(data)
        return self.scaler.fit_transform(x)

    def transform(self, data):
        x = np.sqrt(data)
        return self.scaler.transform(x)

    def inverse_transform(self, data):
        x = self.scaler.inverse_transform(data)
        return np.square(x)