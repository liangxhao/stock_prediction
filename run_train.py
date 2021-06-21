import torch
import numpy as np
import data_prepare
import dataset
import network
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


device = "cuda"
test_days = 60 * 3
batch_size = 64

company_data = data_prepare.read_data("AAPL")
company_data = data_prepare.cutoff_data(company_data)

obj_data = company_data[['Close']].to_numpy()
train_data = obj_data[:-test_days]
test_data = obj_data[-test_days:]

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_norm = scaler.fit_transform(train_data)
test_data_norm = scaler.transform(test_data)

X_train, y_train = dataset.preprocess_lstm(train_data_norm)
X_test, y_test = dataset.preprocess_lstm(test_data_norm)

train_loader = dataset.build_loader(X_train, y_train, batch_size=batch_size)

model = network.LSTMPriceModel(1, 1)
model=model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    avg_loss = 0
    for step, (input, target) in enumerate(train_loader):
        input=input.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        pred, loss, hidden = model(input, target)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

    print(f"epoch: {epoch}, loss: {avg_loss/len(X_train)}")


