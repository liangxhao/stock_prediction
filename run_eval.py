import pandas as pd
import torch
import numpy as np
import data_prepare
import network
import matplotlib.pyplot as plt
import dataset
from transform import Transform
from config import *

company_data = data_prepare.read_data(symbol)
company_data = data_prepare.cutoff_data(company_data)

obj_data = company_data[columns].to_numpy()

train_data = obj_data[:-test_days]
scaler = Transform()
train_data_norm = scaler.fit_transform(train_data)

obj_data_norm = scaler.transform(obj_data)
test_loader = dataset.build_loader(obj_data_norm, batch_size=1, n_steps=n_steps, shuffle=False)

model = network.LSTMPriceModel(obj_data.shape[1], obj_data.shape[1])
print(model)

ckpt_filename = f'{model_path}/model_300.pt'
state_dict = torch.load(ckpt_filename)
model.load_state_dict(state_dict)
model.eval()
model=model.to(device)


################ test ################
targets = []
preds = []
for input, target in test_loader:
    with torch.no_grad():
        model.init_hidden(1, device)
        output, _ = model(input)
    preds.append(output.detach().cpu().numpy())
    targets.append(target.detach().cpu().numpy())

preds = scaler.inverse_transform(np.concatenate(preds, axis=0))[:, -1]
targets = scaler.inverse_transform(np.concatenate(targets, axis=0))[:, -1]

# true
plt.plot(company_data.index[n_steps:], targets)

# pred
train_pos = n_steps
test_pos = len(obj_data_norm) - test_days
plt.plot(company_data.index[train_pos: test_pos], preds[:-test_days])
plt.plot(company_data.index[test_pos:], preds[-test_days:])

plt.show()

