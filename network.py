import torch
import torch.nn as nn


class LSTMPriceModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers = 1):
        super(LSTMPriceModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = 32

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=self.hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.hidden_size, output_size)
        self.hidden_cell = None

        self.criterion = nn.MSELoss()

    def init_hidden(self, batch_size=1, device='cpu'):
        self.hidden_cell = (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device),
                       torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))

    def forward(self, X, y=None):
        output, self.hidden_cell = self.lstm(X, self.hidden_cell)
        pred = self.linear(output[:, -1, :])

        if y is not None:
            loss = self.criterion(pred, y)
        else:
            loss = None

        return pred, loss