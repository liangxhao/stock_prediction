import numpy as np
import torch
import torch.utils.data as data_utils

class Dataset(data_utils.Dataset):
    def __init__(self, data, n_steps):
        super(Dataset, self).__init__()
        self.n_steps = n_steps
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data) - self.n_steps

    def __getitem__(self, idx):
        return self.data[idx:idx+self.n_steps, :], self.data[idx+self.n_steps, :]


def build_loader(data, batch_size, n_steps, shuffle=True):
    return data_utils.DataLoader(
        Dataset(data, n_steps),
        batch_size = batch_size,
        shuffle=shuffle
        )


