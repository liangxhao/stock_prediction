import os
import torch
import numpy as np
import data_prepare
import dataset
import network
import matplotlib.pyplot as plt
from transform import Transform
from sklearn.metrics import mean_absolute_error

from config import *


os.makedirs(model_path, exist_ok=True)

company_data = data_prepare.read_data(symbol)
company_data = data_prepare.cutoff_data(company_data)

obj_data = company_data[columns].to_numpy()
train_data = obj_data[:-test_days]
test_data = obj_data[-test_days:]

scaler = Transform()
train_data_norm = scaler.fit_transform(train_data)
test_data_norm = scaler.transform(test_data)

train_loader = dataset.build_loader(train_data_norm, batch_size=batch_size, n_steps=n_steps)

model = network.LSTMPriceModel(obj_data.shape[1], obj_data.shape[1])
model=model.to(device)
model.train()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.1)

epochs = 5000
for epoch in range(epochs):
    avg_loss = 0
    avg_error = 0
    for step, (input, target) in enumerate(train_loader):
        input=input.to(device)
        target=target.to(device)

        optimizer.zero_grad()
        model.init_hidden(len(input), device)
        pred, loss = model(input, target)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        avg_error += mean_absolute_error(target.view(-1).detach().cpu().numpy(), pred.view(-1).detach().cpu().numpy())

    if epoch > 5 and epoch % 2 == 0:
        torch.save(model.state_dict(), f'{model_path}/model_{epoch}.pt')
    print(f"epoch: {epoch}, loss: {avg_loss/len(train_loader)}, error: {avg_error/len(train_loader)}")


