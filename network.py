import torch
import torch.nn as nn


class LSTMPriceModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMPriceModel, self).__init__()
        self.hidden_size = input_size * 5

        self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.hidden_size, output_size)

        self.criterion = nn.MSELoss()


    def forward(self, X, y=None, hidden_cell=None):
        if hidden_cell is None:
            hidden_cell = (torch.zeros((1, len(X), self.hidden_size), device=X.device),
                            torch.zeros((1, len(X), self.hidden_size), device=X.device))

        output, hidden_cell_next = self.lstm(X, hidden_cell)

        output = self.relu(output[:, -1, :])
        pred = self.linear(output)

        if y is not None:
            loss = self.criterion(pred, y)
        else:
            loss = None

        return pred, loss, hidden_cell_next