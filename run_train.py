# https://www.kaggle.com/thibauthurson/stock-price-prediction-with-lstm-multi-step-lstm
import data_prepare
import dataset
import matplotlib.pyplot as plt


company_data = data_prepare.read_data("AAPL")
company_data = data_prepare.cutoff_data(company_data)

obj_data = company_data['Close']
X, y = dataset.preprocess_lstm(obj_data.to_numpy())
test_days = 60

X_train, y_train = X[:-test_days], y[:-test_days]
X_test, y_test = X[-test_days:], y[-test_days:]

# Plot
# train_original = obj_data.iloc[:-test_days]
# test_original = obj_data.iloc[-test_days:]
# plt.figure(figsize=(10,6))
# plt.grid(True)
# plt.xlabel('Dates')
# plt.ylabel('Closing Prices')
# plt.plot(train_original, 'b', label='Train data')
# plt.plot(test_original, 'g', label='Test data')
# plt.legend()
# plt.show()