import numpy as np
import torch
import torch.utils.data as data_utils

class Dataset(data_utils.Dataset):
    def __init__(self, X, y):
        super(Dataset, self).__init__()

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_loader(X, y, batch_size, shuffle=True):
    return data_utils.DataLoader(
        Dataset(X, y),
        batch_size = batch_size,
        shuffle=shuffle
        )


def preprocess_lstm(sequence, n_steps=60, n_features=1):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix >= len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], n_features))
    return X, y


