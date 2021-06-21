import torch.nn as nn


class LSTMPriceModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMPriceModel, self).__init__()
        hidden_dim = input_size * 2

        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(hidden_dim, output_size)

        self.criterion = nn.MSELoss()


    def forward(self, X, y=None):
        output, (h, c) = self.lstm(X)
        output = self.relu(output)
        pred = self.linear(output)
        loss = self.criterion(pred, y)

        return